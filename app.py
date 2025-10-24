from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import LEDForConditionalGeneration, LEDTokenizerFast, pipeline
import logging
import os
import re
import sys
import subprocess
from collections import Counter, defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)``
CORS(app)

# Global classifier cache
CLASSIFIER = None

# Fast core categories - optimized for speed
DEFAULT_CATEGORIES = [
    "political", "sports", "business", "technology", "health",
    "science", "environmental", "entertainment", "crime"
]

class NewsSummarizer:
    def __init__(self, model_path="lorecast/model_final"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = LEDTokenizerFast.from_pretrained(model_path)

            logger.info("Loading model...")
            self.model = LEDForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def preprocess_text(self, text):
        """Clean and normalize text while preserving numbers and important data."""
        # Replace problematic Unicode characters
        text = text.replace('\u20b1', 'P')  # Philippine peso symbol
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u201c', '"')  # Left double quote
        text = text.replace('\u201d', '"')  # Right double quote
        text = text.replace('\u2018', "'")  # Left single quote
        text = text.replace('\u2019', "'")  # Right single quote

        # Remove multiple spaces between words while preserving numbers
        text = re.sub(r'\s+', ' ', text)

        # Remove extra spaces around punctuation but preserve number formatting
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)

        # Preserve common number formats (e.g., "1,000", "3.14", "$100", "50%")
        text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)  # Fix "1 , 000" -> "1,000"

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def extract_key_numbers(self, text):
        """Extract key numbers and numerical data from text WITH CONTEXT."""
        # Extract various number formats WITH their units/context
        numbers_data = {
            'percentages': re.findall(r'\d+\.?\d*%', text),
            'currency': re.findall(r'[$P£€¥]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T))?', text),
            'large_numbers': re.findall(r'\d+(?:,\d{3})+(?:\.\d+)?', text),
            'dates': re.findall(r'\b(?:19|20)\d{2}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b', text),
            'temperatures': re.findall(r'\d+(?:\.\d+)?\s*(?:degrees?(?:\s*[CF])?|°[CF]?|celsius|fahrenheit)', text, re.IGNORECASE),
            'measurements': re.findall(r'\d+(?:\.\d+)?\s*(?:meters?|km|miles?|feet|inches?|kg|pounds?|liters?|grams?)', text, re.IGNORECASE),
            'speeds': re.findall(r'\d+(?:\.\d+)?\s*(?:mph|km/h|kph)', text, re.IGNORECASE),
            'counts': re.findall(r'\d+(?:,\d{3})*\s+(?:jobs|people|workers|deaths|cases|victims|students|patients)', text, re.IGNORECASE),
            'general_numbers': re.findall(r'\b\d+\.?\d*\b', text)
        }

        # Get top 10 most important numbers (prioritize larger, more specific numbers)
        all_numbers = []
        all_numbers.extend(numbers_data['percentages'][:5])
        all_numbers.extend(numbers_data['currency'][:5])
        all_numbers.extend(numbers_data['temperatures'][:3])
        all_numbers.extend(numbers_data['measurements'][:3])
        all_numbers.extend(numbers_data['speeds'][:2])
        all_numbers.extend(numbers_data['counts'][:5])
        all_numbers.extend(numbers_data['large_numbers'][:3])
        all_numbers.extend(numbers_data['dates'][:2])

        # Remove duplicates while preserving order
        seen = set()
        unique_numbers = []
        for num in all_numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)

        return unique_numbers[:15]  # Return top 15 (increased from 10)

    def extract_number_sentences(self, text):
        """Extract sentences containing important numbers."""
        sentences = re.split(r'[.!?]+', text)
        number_sentences = []

        for sentence in sentences:
            # Find sentences with numbers
            if re.search(r'\d+', sentence):
                # Check if it's an important number (not just dates/times)
                if any(keyword in sentence.lower() for keyword in
                       ['percent', '%', 'billion', 'million', 'workers', 'jobs',
                        'killed', 'injured', 'died', 'pesos', 'dollars', '$',
                        'kidnapped', 'filipino', 'citizens', 'people', 'victims']):
                    number_sentences.append(sentence.strip())

        return number_sentences[:8]  # Top 8 number-heavy sentences

    def extract_key_entities(self, text):
        """Extract key entities (persons, organizations) from text - ENHANCED."""
        if not SPACY_AVAILABLE:
            # ENHANCED regex-based fallback with better entity recognition
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

            # Enhanced filtering for common words and better entity recognition
            common_words = {'The', 'This', 'That', 'House', 'Speaker', 'Deputy', 'According', 'However', 'Office', 'President', 'Minister', 'Secretary', 'Congress', 'Senate', 'Government'}

            # Separate names and potential organizations
            potential_people = []
            potential_orgs = []

            for word in words:
                if word not in common_words and len(word) > 2:
                    # Simple heuristics for classification
                    if any(title in word for title in ['Corporation', 'Company', 'Inc', 'Ltd', 'Foundation', 'Organization', 'Department', 'Ministry', 'Agency']):
                        potential_orgs.append(word)
                    elif len(word.split()) <= 3:  # Names typically 1-3 words
                        potential_people.append(word)
                    else:
                        potential_orgs.append(word)

            return {'PERSON': potential_people[:10], 'ORG': potential_orgs[:10]}

        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)

            # ENHANCED: Extract more entity types and filter better
            people = []
            orgs = []

            for ent in doc.ents:
                if ent.label_ == 'PERSON' and len(ent.text.strip()) > 2:
                    people.append(ent.text.strip())
                elif ent.label_ in ['ORG', 'GPE'] and len(ent.text.strip()) > 2:  # Include geopolitical entities
                    orgs.append(ent.text.strip())

            # Remove duplicates while preserving order
            people = list(dict.fromkeys(people))
            orgs = list(dict.fromkeys(orgs))

            return {'PERSON': people[:10], 'ORG': orgs[:10]}
        except OSError:
            # Fallback if spacy model not installed
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            common_words = {'The', 'This', 'That', 'House', 'Speaker', 'Deputy', 'According', 'However', 'Office'}
            names = [w for w in words if w not in common_words and len(w) > 2]
            return {'PERSON': names[:10], 'ORG': []}

    def analyze_content_complexity(self, article):
        """Analyze article content to determine optimal summary parameters."""
        words = article.split()
        sentences = article.split('.')

        article_length = len(words)
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Count key elements
        entities = self.extract_key_entities(article)
        num_entities = len(entities['PERSON']) + len(entities['ORG'])

        # Count numbers/statistics (indicates data-heavy content)
        numbers = len(re.findall(r'\d+\.?\d*%?', article))

        complexity_score = 0

        # Length factor
        if article_length > 1000:
            complexity_score += 2
        elif article_length > 500:
            complexity_score += 1

        # Entity density factor
        entity_density = num_entities / max(article_length / 100, 1)
        if entity_density > 3:
            complexity_score += 2
        elif entity_density > 1.5:
            complexity_score += 1

        # Data density factor
        data_density = numbers / max(article_length / 100, 1)
        if data_density > 5:
            complexity_score += 2
        elif data_density > 2:
            complexity_score += 1

        # Sentence complexity factor
        if avg_sentence_length > 25:
            complexity_score += 1

        return {
            'complexity_score': complexity_score,
            'article_length': article_length,
            'num_entities': num_entities,
            'entity_density': entity_density,
            'data_density': data_density,
            'avg_sentence_length': avg_sentence_length
        }

    def calculate_dynamic_parameters(self, article):
        """Dynamically calculate optimal generation parameters based on content analysis."""
        analysis = self.analyze_content_complexity(article)

        complexity = analysis['complexity_score']
        article_length = analysis['article_length']

        # Dynamic ratio adjustment - BALANCED for better proportionality
        if complexity >= 5:  # Very complex content
            ratio = 0.25
            beams = 8
            length_penalty = 2.8
        elif complexity >= 3:  # Moderately complex
            ratio = 0.22
            beams = 8
            length_penalty = 2.6
        elif complexity >= 1:  # Simple content
            ratio = 0.20
            beams = 8
            length_penalty = 2.4
        else:  # Very simple content
            ratio = 0.18
            beams = 6
            length_penalty = 2.2

        # Length-based adjustments
        if article_length < 200:
            ratio = max(ratio, 0.25)
        elif article_length > 2000:
            ratio = min(ratio, 0.15)

        target_length = int(article_length * ratio)

        # Apply intelligent bounds
        if article_length < 200:
            target_length = max(target_length, 80)
        elif article_length < 500:
            target_length = max(target_length, 120)
        elif article_length < 1000:
            target_length = max(target_length, 200)
        elif article_length < 2000:
            target_length = max(target_length, 280)
        else:
            target_length = max(target_length, 350)

        target_length = min(target_length, 1000)
        min_length = max(200, int(target_length * 0.75))

        return {
            'target_length': target_length,
            'min_length': min_length,
            'num_beams': beams,
            'length_penalty': length_penalty,
            'ratio_used': ratio,
            'analysis': analysis
        }

    def restore_numbers_in_summary(self, summary, article):
        """ACTIVELY restore numbers that model omitted - inject them back into summary."""

        # Define patterns to extract number+context from article and find orphaned units in summary
        restoration_patterns = [
            # Temperature: "35 degrees C" → summary has "degrees C"
            {
                'article_pattern': r'(\d+(?:\.\d+)?)\s*(degrees?\s*[CF]?|°[CF]?|celsius|fahrenheit)',
                'summary_pattern': r'\b(degrees?\s*[CF]?|celsius|fahrenheit)\b',
                'inject': lambda num, unit: f'{num} {unit}'
            },
            # Percentage: "25%" → summary has "percent"
            {
                'article_pattern': r'(\d+(?:\.\d+)?)\s*(%|percent)',
                'summary_pattern': r'\b(percent|per\s+cent)\b',
                'inject': lambda num, unit: f'{num}%'
            },
            # Currency with scale: "50 billion pesos" → summary has "billion pesos"
            {
                'article_pattern': r'([$P£€¥]?\s*\d+(?:,\d{3})*(?:\.\d+)?)\s+(billion|million|thousand)\s+(pesos?|dollars?|pounds?|euros?)?',
                'summary_pattern': r'\b(billion|million|thousand)\s+(pesos?|dollars?|pounds?|euros?)\b',
                'inject': lambda num, scale, curr: f'{num} {scale} {curr}'
            },
            # Job counts: "15,000 jobs" → summary has "jobs"
            {
                'article_pattern': r'(\d+(?:,\d{3})*)\s+(jobs|people|workers|deaths|cases|victims)',
                'summary_pattern': r'\b(jobs|people|workers|deaths|cases|victims)\b',
                'inject': lambda num, unit: f'{num} {unit}'
            },
            # Measurements: "100 meters" → summary has "meters"
            {
                'article_pattern': r'(\d+(?:\.\d+)?)\s+(meters?|km|kilometers?|miles?|feet)',
                'summary_pattern': r'\b(meters?|km|kilometers?|miles?|feet)\b',
                'inject': lambda num, unit: f'{num} {unit}'
            },
            # Counts: "12 bridges" → summary has "bridges"
            {
                'article_pattern': r'(\d+)\s+(bridges?|highways?|roads?|buildings?|hospitals?|schools?)',
                'summary_pattern': r'\b(bridges?|highways?|roads?|buildings?|hospitals?|schools?)\b',
                'inject': lambda num, unit: f'{num} {unit}'
            },
            # Time periods: "3 years" → summary has "years"
            {
                'article_pattern': r'(\d+)\s+(years?|months?|weeks?|days?|hours?)',
                'summary_pattern': r'\b(years?|months?|weeks?|days?|hours?)\b',
                'inject': lambda num, unit: f'{num} {unit}'
            },
        ]

        original_summary = summary
        injections_made = []

        # Try each restoration pattern
        for pattern_config in restoration_patterns:
            # Find all number+unit combinations in article
            article_matches = re.findall(pattern_config['article_pattern'], article, re.IGNORECASE)

            if not article_matches:
                continue

            # Check if summary has the unit WITHOUT the number
            for match in article_matches:
                # Build the replacement text
                if callable(pattern_config['inject']):
                    try:
                        replacement_text = pattern_config['inject'](*match)
                    except:
                        continue

                    # Find orphaned unit in summary and replace with number+unit
                    if re.search(pattern_config['summary_pattern'], summary, re.IGNORECASE):
                        # Only inject if the full number+unit combo isn't already there
                        if replacement_text.lower() not in summary.lower():
                            summary = re.sub(
                                pattern_config['summary_pattern'],
                                replacement_text,
                                summary,
                                count=1,
                                flags=re.IGNORECASE
                            )
                            injections_made.append(replacement_text)

        if injections_made:
            logger.info(f"Number injections made: {injections_made}")
        else:
            logger.info("No number injections needed")

        return summary

    def remove_hallucinations(self, summary, article):
        """IMPROVED: Detect and remove hallucinated facts not in the original article."""
        # Extract proper nouns (capitalized words) from summary
        summary_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary)
        article_lower = article.lower()

        # Check each entity exists in article
        hallucinated = []
        for entity in set(summary_entities):
            if len(entity) <= 3:  # Skip very short words
                continue
            if entity.lower() not in article_lower:
                # Common words that might not be hallucinations
                common_words = {'The', 'This', 'That', 'These', 'Those', 'According', 'However', 'Officials', 'Government'}
                if entity not in common_words:
                    hallucinated.append(entity)
                    logger.warning(f"Potential hallucination detected: '{entity}' not found in article")

        # Remove sentences containing hallucinations
        if hallucinated:
            sentences = re.split(r'[.!?]+', summary)
            clean_sentences = []
            for sent in sentences:
                has_hallucination = any(h in sent for h in hallucinated)
                if not has_hallucination and sent.strip():
                    clean_sentences.append(sent.strip())
            if clean_sentences:
                summary = '. '.join(clean_sentences)
                if not summary.endswith('.'):
                    summary += '.'
                logger.info(f"Removed {len(hallucinated)} hallucinated entities: {hallucinated}")

        return summary

    def fix_contradictions(self, summary):
        """IMPROVED: Remove contradictory or nonsensical sentences."""
        sentences = re.split(r'[.!?]+', summary)
        clean_sentences = []

        for sent in sentences:
            sent_lower = sent.lower().strip()
            if not sent_lower:
                continue

            # Skip nonsensical patterns
            nonsense_patterns = [
                'unclear if the militants are militants',
                'unclear if the abductors are',
                'it is not clear if the militants',
                'whether or not to vote on whether to vote',
                'whether to vote for it',
                'whether the militants are militants',
                'if the militants are members of',
            ]

            is_nonsense = any(pattern in sent_lower for pattern in nonsense_patterns)
            if is_nonsense:
                logger.info(f"Removed contradictory/nonsensical sentence: {sent[:80]}...")
                continue

            # Check for excessive repetition within sentence
            words = sent.split()
            if len(words) > 5:
                # Count word frequency in sentence
                word_counts = {}
                for word in words:
                    word_lower = word.lower()
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

                # If any word appears more than 3 times in one sentence, it's likely garbled
                max_count = max(word_counts.values()) if word_counts else 0
                if max_count > 3:
                    logger.info(f"Removed repetitive sentence: {sent[:80]}...")
                    continue

            clean_sentences.append(sent.strip())

        if clean_sentences:
            return '. '.join(clean_sentences) + '.'
        return summary

    def post_process_summary(self, summary, article):
        """PHASE 4: Post-process summary for better coverage and coherence."""
        # Remove common prefixes that indicate model artifacts
        summary = re.sub(r'^(NEW:\s*|SUMMARY:\s*|Summary:\s*|IMPORTANT NUMBERS.*?:\s*)', '', summary, flags=re.IGNORECASE)

        # Remove the instruction prompt if it leaked into summary
        summary = re.sub(r'Summarize the article.*?:\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'INCLUDE ALL IMPORTANT NUMBERS:\s*', '', summary, flags=re.IGNORECASE)

        # Split into sentences and process each
        sentences = re.split(r'[.!?]+', summary)
        clean_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 8:  # Skip very short fragments
                continue

            # Check for coherent English text
            words = sentence.split()
            coherent_words = []

            for i, word in enumerate(words):
                # Skip words that are clearly garbled
                if len(word) > 20:  # Extremely long words are likely garbled
                    break
                if not re.search(r'[a-zA-Z]', word):  # Must contain letters
                    continue
                if len(word) > 0 and sum(1 for c in word if c.isalpha()) / len(word) < 0.6:  # At least 60% letters
                    break

                # ENHANCED: Detect repetitive patterns like "and and and"
                if len(coherent_words) >= 2:
                    if word == coherent_words[-1] == coherent_words[-2]:  # Same word repeated 3 times
                        break
                    if word == "and" and coherent_words[-1] == "and":  # Stop at "and and"
                        break

                coherent_words.append(word)

            # Only keep sentences with reasonable content
            if len(coherent_words) >= 5:  # At least 5 coherent words
                clean_sentence = ' '.join(coherent_words)
                # Stop processing if we hit nonsense patterns
                if any(pattern in clean_sentence.lower() for pattern in ['inquiry into', 'click here', 'biggest takeaway', 'stay closed']):
                    break
                clean_sentences.append(clean_sentence)

        # Reconstruct clean summary
        if clean_sentences:
            summary = '. '.join(clean_sentences)
            if not summary.endswith('.'):
                summary += '.'
        else:
            # Fallback: take first 100 words and clean them
            words = summary.split()[:100]
            clean_words = [w for w in words if len(w) <= 15 and re.search(r'[a-zA-Z]', w)]
            summary = ' '.join(clean_words[:50]) + '.'

        # IMPROVED: Remove contradictions and nonsensical sentences
        summary = self.fix_contradictions(summary)

        # IMPROVED: Remove hallucinated facts
        summary = self.remove_hallucinations(summary, article)

        # Restore numbers from article (existing pattern-based restoration)
        summary = self.restore_numbers_in_summary(summary, article)

        # Final cleanup
        summary = re.sub(r'\s+', ' ', summary)  # Remove extra spaces
        summary = re.sub(r'\s*\.\s*', '. ', summary)  # Fix spacing around periods
        summary = summary.strip()

        return summary

    def extract_key_sentences(self, article_text, target_sentences=6):
        """Extract the most important sentences using hybrid scoring with entity awareness."""
        sentences = re.split(r'[.!?]+', article_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if len(sentences) <= target_sentences:
            return ' '.join(sentences)

        # Extract key entities from the entire article
        key_entities = self.extract_key_entities(article_text)
        all_entity_texts = []
        all_entity_texts.extend(key_entities.get('PERSON', []))
        all_entity_texts.extend(key_entities.get('ORG', []))

        scores = defaultdict(float)

        # Position scoring (first/last sentences important)
        for i, sentence in enumerate(sentences):
            if i == 0:
                scores[i] += 3.0
            elif i < len(sentences) * 0.3:
                scores[i] += 1.5

        # Length scoring (optimal 10-30 words)
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                scores[i] += 2.0
            elif word_count < 6:
                scores[i] -= 1.0

        # Entity presence scoring (NEW)
        for i, sentence in enumerate(sentences):
            entity_count = 0
            for entity in all_entity_texts:
                if entity.lower() in sentence.lower():
                    entity_count += 1

            # Boost score based on entity density
            if entity_count >= 3:
                scores[i] += 4.0  # High entity density
            elif entity_count >= 2:
                scores[i] += 2.5  # Medium entity density
            elif entity_count >= 1:
                scores[i] += 1.5  # At least one entity

        # Keyword density scoring
        article_words = article_text.lower().split()
        word_freq = Counter(article_words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        important_words = [word for word, freq in word_freq.most_common(15) if word not in stop_words and len(word) > 3]

        for i, sentence in enumerate(sentences):
            sentence_words = sentence.lower().split()
            keyword_count = sum(1 for word in sentence_words if word in important_words)
            scores[i] += keyword_count * 0.5

        # Entity and number scoring
        for i, sentence in enumerate(sentences):
            # Capitalized words (entities)
            capitalized = len([word for word in sentence.split() if word[0].isupper() and len(word) > 1])
            scores[i] += capitalized * 0.3

            # Numbers and percentages
            numbers = len(re.findall(r'\d+\.?\d*%?', sentence))
            scores[i] += numbers * 0.5

            # Quotes (important statements)
            if '"' in sentence or "'" in sentence:
                scores[i] += 1.0

        # Get top sentences
        sentence_scores = [(i, scores[i], sentence) for i, sentence in enumerate(sentences)]
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:target_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Maintain order

        return '. '.join([sentence for _, _, sentence in top_sentences]) + '.'

    def summarize(self, text, mode="smart", use_dynamic=True):
        try:
            # Preprocess the text
            cleaned_article = self.preprocess_text(text)

            # HYBRID APPROACH: Use extractive pre-filtering for longer articles
            original_words = len(text.split())
            if original_words > 500:
                # Extract key sentences first (extractive phase)
                extracted_content = self.extract_key_sentences(text, target_sentences=8)
                extracted_words = len(extracted_content.split())
                logger.info(f"Hybrid mode: {original_words} -> {extracted_words} words ({extracted_words/original_words:.1%} compression)")

                # Use extracted content for abstractive processing
                cleaned_article = self.preprocess_text(extracted_content)
            else:
                logger.info(f"Direct mode: Using full article ({original_words} words)")

            # Extract entities for enhanced processing
            entities = self.extract_key_entities(text)
            people = entities.get('PERSON', [])
            organizations = entities.get('ORG', [])

            # Extract key numbers from the article
            key_numbers = self.extract_key_numbers(text)

            logger.info(f"Extracted entities: {len(people)} people, {len(organizations)} organizations, {len(key_numbers)} key numbers")

            # IMPROVED NUMBER-FOCUSED PROMPTING with stronger instructions
            if use_dynamic and key_numbers:
                # Extract sentences with important numbers for context
                number_sentences = self.extract_number_sentences(text)

                # Create strong instruction-based prompt
                number_instruction = f"IMPORTANT NUMBERS TO PRESERVE: {', '.join(key_numbers[:10])}\n\n"

                if number_sentences:
                    number_instruction += "KEY FACTS WITH NUMBERS:\n"
                    for sent in number_sentences:
                        number_instruction += f"- {sent}\n"
                    number_instruction += "\n"

                number_instruction += "Summarize the article below and INCLUDE ALL IMPORTANT NUMBERS:\n\n"

                cleaned_article = number_instruction + cleaned_article
                logger.info(f"IMPROVED number preservation prompt added with {len(key_numbers)} numbers and {len(number_sentences)} fact sentences")

            logger.info(f"Extracted for post-processing: {len(people)} people, {len(organizations)} organizations, {len(key_numbers)} key numbers")

            # Enhanced parameter optimization for quality
            if use_dynamic:
                dynamic_params = self.calculate_dynamic_parameters(text)
                article_length = dynamic_params['analysis']['article_length']

                # BALANCED: Scale max_tokens conservatively based on article length
                base_max_tokens = dynamic_params['target_length']
                # More conservative max_tokens for shorter articles
                if article_length < 500:
                    max_tokens = min(base_max_tokens, 150)  # Cap short articles at 150 words
                elif article_length < 1000:
                    max_tokens = min(base_max_tokens, 250)  # Cap medium articles at 250 words
                else:
                    max_tokens = max(base_max_tokens, 300)  # Allow longer for long articles

                # BALANCED: Smart minimum length scaling with reasonable caps
                if article_length < 200:  # Very short articles
                    min_length = max(40, min(80, int(max_tokens * 0.7)))  # Cap at 80 words max
                elif article_length < 500:  # Short articles
                    min_length = max(50, min(100, int(max_tokens * 0.7)))  # Cap at 100 words max
                elif article_length < 1000:  # Medium articles
                    min_length = max(80, min(150, int(max_tokens * 0.6)))  # Cap at 150 words max
                else:  # Longer articles - apply reasonable minimum
                    min_length = max(120, min(200, int(max_tokens * 0.6)))  # Cap at 200 words max

                # BALANCED parameters for accuracy
                num_beams = dynamic_params['num_beams']  # Use full beam width
                length_penalty = 2.0  # REDUCED: Less aggressive to allow more detail

                logger.info(f"PHASE 1 ENHANCEMENT: Aggressive parameter optimization active")
                logger.info(f"Dynamic optimization: Article={dynamic_params['analysis']['article_length']} words")
                logger.info(f"Complexity score: {dynamic_params['analysis']['complexity_score']}/7")
                logger.info(f"Enhanced parameters: ratio={dynamic_params['ratio_used']:.2f}, beams={num_beams}, penalty={length_penalty}")
                logger.info(f"Target summary: {max_tokens} words (min={min_length} - ENHANCED for better coverage)")
            else:
                # Quality-focused defaults
                max_tokens = 150
                min_length = 60
                num_beams = 5
                length_penalty = 1.8

            # Tokenize input text
            inputs = self.tokenizer(
                cleaned_article,
                return_tensors="pt",
                max_length=4096,
                truncation=True,
                padding=True
            ).to(self.device)

            # Generate summary with IMPROVED parameters for accuracy
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=11,  # IMPROVED: Increased from 8 for better quality
                    no_repeat_ngram_size=6,  # IMPROVED: Increased from 3 to prevent more repetition
                    length_penalty=1.2,  # IMPROVED: Reduced from 2.0 to allow more detail
                    early_stopping=True,  # ACCURACY: Stop when complete to prevent hallucination
                    max_new_tokens=max_tokens,
                    min_length=min_length,
                    do_sample=False,  # ACCURACY: Deterministic beam search for factual consistency
                    repetition_penalty=1.7,  # IMPROVED: Increased from 1.2 for stronger anti-repetition
                    forced_bos_token_id=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    encoder_no_repeat_ngram_size=5,  # IMPROVED: Increased from 2 to prevent input copying
                )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.info(f"RAW MODEL OUTPUT: {summary}")

            # DISABLED: Post-process for better quality
            # if use_dynamic:
            #     summary = self.post_process_summary(summary, text)
            #     logger.info("PHASE 4 ENHANCEMENT: Content coverage optimization applied")
            #     logger.info(f"POST-PROCESSED OUTPUT: {summary}")

            logger.info("POST-PROCESSING DISABLED - Returning raw model output only")

            # DISABLED: Post-process for better quality
            # if use_dynamic:
            #     summary = self.post_process_summary(summary, text)
            #     logger.info("PHASE 4 ENHANCEMENT: Content coverage optimization applied")
            #     logger.info(f"POST-PROCESSED OUTPUT: {summary}")

            logger.info("POST-PROCESSING DISABLED - Returning raw model output only")

            # Perform article classification - INTEGRATED with GPU support and caching
            article_category = self.classify_article_fast(text)
            logger.info(f"Article classified as: {article_category}")

            # Return result with summary and category
            result = {
                "summary": summary,
                "category": article_category
            }

            return result

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise e

    def calculate_coverage_score(self, article, summary):
        """Calculate what percentage of key entities are covered in summary."""
        article_entities = self.extract_key_entities(article)

        total_entities = len(article_entities['PERSON']) + len(article_entities['ORG'])
        if total_entities == 0:
            return 1.0

        covered = 0
        for person in article_entities['PERSON']:
            if person in summary:
                covered += 1
        for org in article_entities['ORG']:
            if org in summary:
                covered += 1

        return covered / total_entities

    def fallback_classify(self, text, categories):
        """Simple keyword-based classification fallback - EXACT COPY from classify_article.py"""
        text_lower = text.lower()
        scores = {}

        # Comprehensive keywords for each category - ENHANCED
        keywords = {
            # Core topics
            "political": ["government", "president", "congress", "election", "vote", "senator", "representative", "speaker", "minister", "parliament", "policy", "administration", "legislature", "political", "politics", "prime minister", "diplomat", "diplomatic", "treaty", "state department", "foreign minister", "cabinet", "legislation", "sovereignty", "territorial", "jurisdiction", "accession", "ratification", "peace talks", "negotiations", "bilateral", "multilateral", "geopolitical", "state party", "governor", "mayor", "referendum", "coalition", "opposition", "ruling party"],
            "sports": ["game", "team", "player", "score", "match", "championship", "league", "tournament", "coach", "stadium", "athlete", "competition", "olympics", "win", "won", "loss", "defeat", "victory", "season", "playoff", "finals", "goal", "points", "quarterback", "pitcher", "striker", "defender", "midfielder", "baseball", "basketball", "football", "soccer", "tennis", "golf", "racing", "boxing", "swimming", "games", "players", "competed", "champion", "sports", "sporting event", "athletic", "contestants", "delegations", "medals", "prizes", "winning", "compete", "skill", "talents"],
            "business": ["company", "market", "stock", "profit", "revenue", "investment", "financial", "trade", "corporate", "ceo", "industry", "economic", "business", "economy", "earnings", "shares", "shareholders", "merger", "acquisition", "bankruptcy", "fiscal", "quarterly", "annual report", "wall street", "nasdaq", "dow jones", "trading", "commodities", "export", "import", "manufacturing", "retail", "sales", "venture capital"],
            "technology": ["tech", "software", "computer", "digital", "internet", "ai", "artificial intelligence", "app", "platform", "innovation", "startup", "algorithm", "programming", "silicon valley", "smartphone", "tablet", "laptop", "hardware", "cybersecurity", "cloud computing", "blockchain", "cryptocurrency", "bitcoin", "machine learning", "robotics", "5g", "wireless", "semiconductor", "processor", "operating system", "android", "ios", "windows", "linux"],
            "health": ["health", "medical", "hospital", "doctor", "patient", "disease", "treatment", "medicine", "vaccine", "healthcare", "clinic", "therapy", "illness", "syndrome", "infection", "outbreak", "epidemic", "pandemic", "symptoms", "diagnosis", "fever", "virus", "bacterial", "die", "died", "death", "mortality", "sick", "typhus", "plague", "malaria", "cancer", "diabetes", "heart disease", "stroke", "pneumonia", "tuberculosis", "mental health", "surgery", "pharmaceutical", "nursing", "emergency room", "infectious disease", "pathogen", "contagious", "immunization", "medication", "prescription", "medical care", "health crisis"],
            "science": ["research", "study", "scientist", "discovery", "experiment", "laboratory", "scientific", "biology", "physics", "chemistry", "findings", "analysis", "researchers", "peer review", "journal", "publication", "hypothesis", "theory", "observation", "data analysis", "clinical trial", "methodology", "genome", "dna", "molecular", "quantum", "particle", "astronomy", "geology", "neuroscience", "breakthrough", "innovation"],
            "entertainment": ["movie", "film", "music", "celebrity", "actor", "singer", "show", "concert", "album", "theater", "hollywood", "artist", "actress", "director", "producer", "performance", "premiere", "box office", "streaming", "netflix", "spotify", "grammy", "oscar", "emmy", "awards", "television", "tv series", "drama", "comedy", "band", "musician", "entertainment industry", "red carpet", "blockbuster"],
            "education": ["school", "university", "student", "teacher", "education", "learning", "academic", "college", "curriculum", "graduation", "classroom", "professor", "tuition", "campus", "enrollment", "degree", "bachelor", "master", "doctorate", "phd", "undergraduate", "graduate", "scholarship", "exam", "test scores", "literacy", "educational", "pedagogy", "faculty"],
            "crime": ["police", "arrest", "criminal", "court", "trial", "sentence", "investigation", "law enforcement", "crime", "justice", "lawyer", "judge", "murder", "robbery", "theft", "assault", "fraud", "prosecutor", "defendant", "guilty", "innocent", "verdict", "prison", "jail", "felony", "misdemeanor", "detective", "forensic", "evidence", "witness", "testimony", "indictment", "conviction"],
            "international": ["country", "nation", "international", "foreign", "diplomatic", "embassy", "border", "global", "worldwide", "treaty", "relations", "united nations", "nato", "eu", "european union", "middle east", "asia", "africa", "latin america", "sanctions", "trade agreement", "ambassador", "summit", "conference", "alliance", "cooperation"],

            # Specialized topics - ENHANCED
            "environmental": ["climate", "environment", "pollution", "carbon", "green", "sustainability", "renewable", "emissions", "conservation", "ecosystem", "global warming", "climate change", "greenhouse gas", "carbon footprint", "deforestation", "renewable energy", "solar power", "wind energy", "environmental protection", "recycling", "waste management", "air quality", "water quality"],
            "wildlife": ["animals", "species", "wildlife", "nature", "habitat", "endangered", "conservation", "biodiversity", "ecosystem", "fauna", "flora", "endangered species", "national park", "wildlife refuge", "animal population", "extinction", "breeding program", "poaching", "habitat loss"],
            "pollution": ["mercury", "toxic", "contamination", "chemical", "waste", "pollutant", "hazardous", "poison", "cleanup", "environmental damage", "air pollution", "water pollution", "soil contamination", "industrial waste", "sewage", "smog", "acid rain"],
            "conservation": ["protect", "preserve", "endangered", "habitat", "ecosystem", "biodiversity", "restoration", "sanctuary", "reserve", "sustainability", "wildlife conservation", "nature preserve", "protected area", "environmental stewardship"],
            "research": ["study", "analysis", "findings", "data", "investigation", "survey", "report", "evidence", "methodology", "results", "conclusion", "peer reviewed", "research paper", "academic study", "empirical data", "statistical analysis"],
            "medical": ["patient", "treatment", "diagnosis", "therapy", "clinical", "symptoms", "cure", "medication", "surgical", "medical care", "disease", "illness", "infection", "outbreak", "epidemic", "died", "death", "mortality", "fever", "typhus", "virus", "bacterial", "medical emergency", "intensive care", "pathology", "radiology"],
            "military": ["army", "navy", "air force", "defense", "soldier", "war", "combat", "military", "troops", "weapons", "security", "armed forces", "marines", "veterans", "deployment", "warfare", "battalion", "regiment", "pentagon", "military operation", "defense budget", "national security"],
            "agriculture": ["farming", "crops", "farmers", "livestock", "agricultural", "harvest", "rural", "food production", "farming", "cultivation", "agriculture", "irrigation", "pesticides", "fertilizer", "farmland", "cattle", "poultry", "grain", "wheat", "corn", "rice", "farm subsidies"],
            "energy": ["power", "electricity", "oil", "gas", "renewable", "solar", "wind", "nuclear", "energy", "fuel", "electricity", "power plant", "energy production", "petroleum", "coal", "natural gas", "hydroelectric", "geothermal", "energy sector", "power grid", "utility"],
            "transportation": ["traffic", "highway", "road", "vehicle", "transportation", "infrastructure", "construction", "bridge", "tunnel", "transit", "automobile", "railway", "railroad", "aviation", "airport", "shipping", "maritime", "public transportation", "subway", "bus", "train"],
            "space": ["space", "nasa", "satellite", "rocket", "mars", "moon", "astronaut", "cosmic", "universe", "galaxy", "planetary", "space station", "launch", "orbit", "spacecraft", "apollo", "international space station", "spacex", "astronomical", "telescope"],
            "disaster": ["disaster", "emergency", "crisis", "accident", "fire", "flood", "earthquake", "storm", "hurricane", "rescue", "evacuation", "natural disaster", "tornado", "tsunami", "wildfire", "emergency response", "casualties", "damage", "relief efforts", "survivors"],
            "immigration": ["immigrant", "refugee", "border", "migration", "asylum", "citizenship", "visa", "deportation", "immigration", "migrant", "immigration policy", "border control", "illegal immigration", "green card", "naturalization", "immigration reform", "refugee crisis"]
        }

        for category in categories:
            if category.lower() in keywords:
                keyword_list = keywords[category.lower()]
                # Count matches with weighted scoring
                total_score = 0
                for keyword in keyword_list:
                    # Multi-word keywords get exact phrase matching
                    if ' ' in keyword:
                        if keyword in text_lower:
                            total_score += 2  # Higher weight for exact phrase match
                    else:
                        # Single word keywords use word boundary matching
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                            total_score += 1

                scores[category] = total_score / len(keyword_list)  # Normalize by number of keywords
            else:
                scores[category] = 0

        # Context-aware boosting: Detect strong category indicators
        # Sports events organized by government should still be classified as sports
        sports_indicators = [
            'sports commission', 'games', 'competition', 'champion', 'players competed',
            'overall champion', 'emerged as', 'won', 'medal', 'tournament', 'playoff'
        ]
        political_indicators = [
            'election', 'voted', 'legislation', 'bill', 'senate', 'congress session',
            'political party', 'campaign', 'referendum'
        ]

        sports_signal = sum(1 for indicator in sports_indicators if indicator in text_lower)
        political_signal = sum(1 for indicator in political_indicators if indicator in text_lower)

        # Apply context boost
        if 'sports' in scores and sports_signal >= 3:
            scores['sports'] *= 1.3  # Boost sports by 30% if strong sports context
        if 'political' in scores and political_signal >= 2:
            scores['political'] *= 1.2  # Boost political by 20% if strong political context

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Convert to expected format
        result = {
            'labels': [item[0] for item in sorted_scores],
            'scores': [item[1] for item in sorted_scores]
        }

        return result

    def load_classifier(self):
        """Load the zero-shot classification model with GPU support and caching."""
        global CLASSIFIER
        if CLASSIFIER is None:
            logger.info("Loading zero-shot classification model...")
            try:
                # Use GPU if available (device 0 for cuda, -1 for cpu)
                device = 0 if torch.cuda.is_available() else -1
                CLASSIFIER = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device
                )
                logger.info(f"Classification model loaded successfully on {'GPU' if device == 0 else 'CPU'}!")
            except Exception as e:
                logger.error(f"Error loading classification model: {e}")
                logger.info("Falling back to keyword-based classification...")
                CLASSIFIER = "fallback"
        return CLASSIFIER

    def classify_article_fast(self, text, categories=None, threshold=0.05):
        """Fast classification - keyword first, then AI if needed."""
        if categories is None:
            categories = DEFAULT_CATEGORIES

        # Try fast keyword matching first
        keyword_result = self.fallback_classify(text, categories)
        if keyword_result['scores'][0] >= 0.1:  # Good keyword match
            return keyword_result['labels'][0]

        # Only use AI model if keyword matching is weak
        classifier = self.load_classifier()

        if classifier == "fallback":
            return keyword_result['labels'][0]

        try:
            # Truncate text for speed
            max_length = 500
            words = text.split()
            if len(words) > max_length:
                text = ' '.join(words[:max_length])

            result = classifier(text, categories)

            # Take first result above threshold
            for label, score in zip(result['labels'], result['scores']):
                if score >= threshold:
                    return label

            # Fallback to keyword result
            return keyword_result['labels'][0]

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return keyword_result['labels'][0]


# Initialize the summarizer (this will load the model at startup)
summarizer = None

def load_model():
    global summarizer
    try:
        summarizer = NewsSummarizer()
        logger.info("Summarizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize summarizer: {str(e)}")
        summarizer = None

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    try:
        if summarizer is None:
            logger.error("Model not loaded")
            return jsonify({"error": "Model not loaded. Please try again later."}), 500

        # Debug: Log the raw request
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request data: {request.data}")

        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")

        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data provided"}), 400

        if 'text' not in data:
            logger.error(f"No 'text' field in data. Available fields: {list(data.keys()) if data else []}")
            return jsonify({"error": "No text field provided"}), 400

        text = data['text'].strip()
        logger.info(f"Received text: '{text[:100]}...' (length: {len(text)})")

        if not text:
            logger.error("Text is empty after stripping")
            return jsonify({"error": "Text cannot be empty"}), 400

        if len(text) < 50:
            logger.error(f"Text too short: {len(text)} characters")
            return jsonify({"error": "Text too short. Please provide at least 50 characters."}), 400

        # Get mode from request (default to smart mode)
        mode = data.get('mode', 'smart')
        use_dynamic = mode in ['smart', 'advanced']

        logger.info(f"Processing text of length: {len(text)} with mode: {mode}")

        result = summarizer.summarize(text, mode=mode, use_dynamic=use_dynamic)
        logger.info(f"Summarization successful. Summary length: {len(result['summary'].split())} words")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if summarizer is not None else "model_not_loaded"
    return jsonify({"status": status})

@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available summarization modes and their descriptions."""
    modes = {
        "basic": {
            "description": "Simple summarization with standard parameters",
            "features": ["Basic summarization", "Category classification"]
        },
        "smart": {
            "description": "Intelligent summarization with dynamic optimization (recommended)",
            "features": [
                "Dynamic parameter optimization",
                "Entity-aware prompting",
                "Content complexity analysis",
                "Enhanced post-processing",
                "Quality metrics"
            ]
        },
        "advanced": {
            "description": "Full-featured mode with all enhancements",
            "features": [
                "All smart mode features",
                "Detailed analysis metrics",
                "Entity coverage tracking",
                "Complexity scoring"
            ]
        }
    }
    return jsonify({"modes": modes, "default": "smart"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "LORECAST News Summarizer API",
        "version": "2.0 - Enhanced with Dynamic Optimization",
        "endpoints": {
            "/api/summarize": "POST - Summarize article text",
            "/api/health": "GET - Check API health",
            "/api/modes": "GET - Get available summarization modes"
        },
        "features": [
            "Dynamic parameter optimization",
            "Entity-aware summarization",
            "Content complexity analysis",
            "Multi-mode operation",
            "Quality metrics"
        ]
    })

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)