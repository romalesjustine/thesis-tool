from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import LEDForConditionalGeneration, LEDTokenizerFast
import logging
import os
import re
from collections import Counter, defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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
            logger.info(f"Generated summary: {summary}")

            # Enhanced category classification
            category = self.classify_category(text.lower())

            # Return simplified result with just summary and category
            result = {
                "summary": summary,
                "category": category
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

    def classify_category(self, text):
        """Keyword-based category classification with scoring - matches classify_article.py implementation"""
        text_lower = text.lower()

        # Comprehensive keywords for each category - EXACTLY from classify_article.py
        keywords = {
            # Core topics
            "political": ["government", "president", "congress", "election", "vote", "senator", "representative", "speaker", "minister", "parliament", "policy", "administration", "legislature"],
            "sports": ["game", "team", "player", "score", "match", "championship", "league", "tournament", "coach", "stadium", "athlete", "competition", "olympics"],
            "business": ["company", "market", "stock", "profit", "revenue", "investment", "financial", "trade", "corporate", "ceo", "industry", "economic"],
            "technology": ["tech", "software", "computer", "digital", "internet", "ai", "artificial intelligence", "data", "app", "platform", "innovation", "startup"],
            "health": ["health", "medical", "hospital", "doctor", "patient", "disease", "treatment", "medicine", "vaccine", "healthcare", "clinic", "therapy"],
            "science": ["research", "study", "scientist", "discovery", "experiment", "laboratory", "scientific", "biology", "physics", "chemistry", "findings", "analysis"],
            "entertainment": ["movie", "film", "music", "celebrity", "actor", "singer", "show", "concert", "album", "theater", "hollywood", "artist"],
            "education": ["school", "university", "student", "teacher", "education", "learning", "academic", "college", "curriculum", "graduation", "classroom"],
            "crime": ["police", "arrest", "criminal", "court", "trial", "sentence", "investigation", "law enforcement", "crime", "justice", "lawyer", "judge"],
            "international": ["country", "nation", "international", "foreign", "diplomatic", "embassy", "border", "global", "worldwide", "treaty", "relations"],

            # Specialized topics
            "environmental": ["climate", "environment", "pollution", "carbon", "green", "sustainability", "renewable", "emissions", "conservation", "ecosystem", "global warming"],
            "wildlife": ["animals", "species", "wildlife", "nature", "habitat", "endangered", "conservation", "biodiversity", "ecosystem", "fauna", "flora"],
            "pollution": ["mercury", "toxic", "contamination", "chemical", "waste", "pollutant", "hazardous", "poison", "cleanup", "environmental damage"],
            "conservation": ["protect", "preserve", "endangered", "habitat", "ecosystem", "biodiversity", "restoration", "sanctuary", "reserve", "sustainability"],
            "research": ["study", "analysis", "findings", "data", "investigation", "survey", "report", "evidence", "methodology", "results", "conclusion"],
            "medical": ["patient", "treatment", "diagnosis", "therapy", "clinical", "symptoms", "cure", "medication", "surgical", "medical care"],
            "military": ["army", "navy", "air force", "defense", "soldier", "war", "combat", "military", "troops", "weapons", "security"],
            "agriculture": ["farming", "crops", "farmers", "livestock", "agricultural", "harvest", "rural", "food production", "farming", "cultivation"],
            "energy": ["power", "electricity", "oil", "gas", "renewable", "solar", "wind", "nuclear", "energy", "fuel", "electricity"],
            "transportation": ["traffic", "highway", "road", "vehicle", "transportation", "infrastructure", "construction", "bridge", "tunnel", "transit"],
            "space": ["space", "nasa", "satellite", "rocket", "mars", "moon", "astronaut", "cosmic", "universe", "galaxy", "planetary"],
            "disaster": ["disaster", "emergency", "crisis", "accident", "fire", "flood", "earthquake", "storm", "hurricane", "rescue", "evacuation"],
            "immigration": ["immigrant", "refugee", "border", "migration", "asylum", "citizenship", "visa", "deportation", "immigration", "migrant"]
        }

        # Fast core categories for common articles
        DEFAULT_CATEGORIES = [
            "political", "sports", "business", "technology", "health",
            "science", "environmental", "entertainment", "crime"
        ]

        scores = {}

        # Calculate scores for all categories
        for category in DEFAULT_CATEGORIES:
            if category in keywords:
                keyword_list = keywords[category]
                score = sum(1 for keyword in keyword_list if keyword in text_lower)
                scores[category] = score / len(keyword_list)  # Normalize by number of keywords
            else:
                scores[category] = 0

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return the category with highest score if above threshold (0.05 minimum)
        if sorted_scores and sorted_scores[0][1] >= 0.05:
            # Capitalize first letter for display
            return sorted_scores[0][0].capitalize()

        return "General"

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