#!/usr/bin/env python
"""
evaluate_custom.py

Flexible evaluation script that can summarize any input article using your RL checkpoint model.
Supports multiple input methods: file, direct text, or stdin.

Usage examples:
  python evaluate_custom.py --text "Your article text here..."
  python evaluate_custom.py --file article.txt
  echo "Article text" | python evaluate_custom.py --stdin
"""

import argparse
import sys
import re
from transformers import LEDForConditionalGeneration, LEDTokenizerFast
from rouge_score import rouge_scorer
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# ─── CONFIGURE HERE ─────────────────────────────────────────────────────────
CHECKPOINT_DIR = "lorecast/model_final2"  # path to your RL model

def load_model(checkpoint_dir):
    """Load the RL checkpoint model and tokenizer."""
    print(f"Loading model from: {checkpoint_dir}")
    tokenizer = LEDTokenizerFast.from_pretrained(checkpoint_dir)
    model = LEDForConditionalGeneration.from_pretrained(checkpoint_dir)
    print("Model loaded successfully!")
    return tokenizer, model

def preprocess_text(text):
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

def extract_key_numbers(text):
    """Extract key numbers and numerical data from text."""
    import re

    # Extract various number formats
    numbers_data = {
        'percentages': re.findall(r'\d+\.?\d*%', text),
        'currency': re.findall(r'[$P£€¥]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T))?', text),
        'large_numbers': re.findall(r'\d+(?:,\d{3})+(?:\.\d+)?', text),
        'dates': re.findall(r'\b(?:19|20)\d{2}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b', text),
        'general_numbers': re.findall(r'\b\d+\.?\d*\b', text)
    }

    # Get top 10 most important numbers (prioritize larger, more specific numbers)
    all_numbers = []
    all_numbers.extend(numbers_data['percentages'][:5])
    all_numbers.extend(numbers_data['currency'][:5])
    all_numbers.extend(numbers_data['large_numbers'][:5])
    all_numbers.extend(numbers_data['dates'][:3])

    # Remove duplicates while preserving order
    seen = set()
    unique_numbers = []
    for num in all_numbers:
        if num not in seen:
            seen.add(num)
            unique_numbers.append(num)

    return unique_numbers[:10]  # Return top 10

def extract_key_entities(text):
    """Extract key entities (persons, organizations) from text - PHASE 2 ENHANCED."""
    if not SPACY_AVAILABLE:
        # ENHANCED regex-based fallback with better entity recognition
        import re
        # Look for capitalized sequences (names, organizations)
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
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        common_words = {'The', 'This', 'That', 'House', 'Speaker', 'Deputy', 'According', 'However', 'Office'}
        names = [w for w in words if w not in common_words and len(w) > 2]
        return {'PERSON': names[:10], 'ORG': []}

def calculate_coverage_score(article, summary):
    """Calculate what percentage of key entities are covered in summary."""
    article_entities = extract_key_entities(article)
    summary_entities = extract_key_entities(summary)

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

def extract_key_sentences(article_text, target_sentences=6):
    """Extract the most important sentences using hybrid scoring."""
    import re
    from collections import Counter, defaultdict

    # Split into sentences
    sentences = re.split(r'[.!?]+', article_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) <= target_sentences:
        return ' '.join(sentences)

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

def analyze_content_complexity(article):
    """Analyze article content to determine optimal summary parameters."""
    words = article.split()
    sentences = article.split('.')

    # Content metrics
    article_length = len(words)
    avg_sentence_length = len(words) / max(len(sentences), 1)

    # Count key elements
    entities = extract_key_entities(article)
    num_entities = len(entities['PERSON']) + len(entities['ORG'])

    # Count numbers/statistics (indicates data-heavy content)
    import re
    numbers = len(re.findall(r'\d+\.?\d*%?', article))

    # Determine content complexity
    complexity_score = 0

    # Length factor
    if article_length > 1000:
        complexity_score += 2
    elif article_length > 500:
        complexity_score += 1

    # Entity density factor
    entity_density = num_entities / max(article_length / 100, 1)  # entities per 100 words
    if entity_density > 3:
        complexity_score += 2
    elif entity_density > 1.5:
        complexity_score += 1

    # Data density factor
    data_density = numbers / max(article_length / 100, 1)  # numbers per 100 words
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

def calculate_dynamic_parameters(article):
    """Dynamically calculate optimal generation parameters based on content analysis."""
    analysis = analyze_content_complexity(article)

    # Base parameters
    base_ratio = 0.15
    base_beams = 8
    base_length_penalty = 1.5

    # Adjust based on complexity
    complexity = analysis['complexity_score']
    article_length = analysis['article_length']

    # Dynamic ratio adjustment - BALANCED for better proportionality
    if complexity >= 5:  # Very complex content
        ratio = 0.25  # BALANCED: Reduced from 0.35 for better proportionality
        beams = 8
        length_penalty = 2.8  # Keep aggressive length penalty
    elif complexity >= 3:  # Moderately complex
        ratio = 0.22  # BALANCED: Reduced from 0.30
        beams = 8
        length_penalty = 2.6  # Keep aggressive length penalty
    elif complexity >= 1:  # Simple content
        ratio = 0.20  # BALANCED: Reduced from 0.28
        beams = 8
        length_penalty = 2.4  # Keep aggressive length penalty
    else:  # Very simple content
        ratio = 0.18  # BALANCED: Reduced from 0.25
        beams = 6
        length_penalty = 2.2  # Keep aggressive length penalty

    # Length-based adjustments
    if article_length < 200:
        ratio = max(ratio, 0.25)  # Ensure minimum coverage for short articles
    elif article_length > 2000:
        ratio = min(ratio, 0.15)  # Cap ratio for very long articles

    # Calculate target length
    target_length = int(article_length * ratio)

    # Apply intelligent bounds with PHASE 1 aggressive minimums
    if article_length < 200:
        target_length = max(target_length, 80)  # ENHANCED: Higher minimum
    elif article_length < 500:
        target_length = max(target_length, 120)  # ENHANCED: Higher minimum
    elif article_length < 1000:
        target_length = max(target_length, 200)  # ENHANCED: Higher minimum
    elif article_length < 2000:
        target_length = max(target_length, 280)  # ENHANCED: Higher minimum
    else:
        target_length = max(target_length, 350)  # ENHANCED: Higher minimum

    # Cap maximum length with PHASE 1 enhancements
    target_length = min(target_length, 1000)  # ENHANCED: Higher maximum
    min_length = max(200, int(target_length * 0.75))  # ENHANCED: Much higher minimum (200+ words)

    return {
        'target_length': target_length,
        'min_length': min_length,
        'num_beams': beams,
        'length_penalty': length_penalty,
        'ratio_used': ratio,
        'analysis': analysis
    }

def calculate_adaptive_length(article, args=None):
    """Calculate adaptive summary length based on article length."""
    article_words = len(article.split())

    # Use adaptive bounds if provided
    if args:
        summary_ratio = getattr(args, 'summary_ratio', 0.15)
        min_words = getattr(args, 'adaptive_min', 50)
        max_words = getattr(args, 'adaptive_max', 1024)
    else:
        summary_ratio = 0.15
        min_words = 50
        max_words = 1024

    # Calculate proportional length
    target_length = int(article_words * summary_ratio)

    # Apply bounds
    target_length = max(min_words, min(target_length, max_words))

    # Adjust based on article size categories with better scaling
    if article_words < 200:  # Short articles (tweets, headlines)
        target_length = max(target_length, 50)
    elif article_words < 500:  # Medium articles (news briefs)
        target_length = max(target_length, 75)
    elif article_words < 1000:  # Long articles (standard news)
        target_length = max(target_length, 120)
    elif article_words < 2000:  # Very long articles (features)
        target_length = max(target_length, 200)
    else:  # Extra long articles (in-depth reports)
        target_length = max(target_length, 300)

    return target_length, article_words

def restore_numbers_in_summary(summary, article):
    """Restore all numbers from article that were converted to words in summary."""
    import re

    # Extract all numbers from article with their context
    number_mappings = []

    # Find percentages (e.g., "50%", "3.5%")
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(%|percent)', article, re.IGNORECASE):
        number = match.group(1)
        number_mappings.append({
            'original': f"{number}%",
            'patterns': [
                r'\b' + re.escape(number) + r'\s*(?:percent|per cent)\b',
                r'\b(?:percent|per cent)\b'
            ]
        })

    # Find currency (e.g., "$1,000", "P500 million")
    for match in re.finditer(r'([$P£€¥]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?)', article):
        number_mappings.append({
            'original': match.group(1),
            'patterns': []
        })

    # Find all standalone numbers
    for match in re.finditer(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', article):
        number = match.group(1)
        number_mappings.append({
            'original': number,
            'patterns': []
        })

    # Find numbers with scale words (e.g., "500 million", "2.5 billion")
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s+(billion|million|thousand|trillion)\b', article, re.IGNORECASE):
        full_number = f"{match.group(1)} {match.group(2)}"
        number_mappings.append({
            'original': full_number,
            'patterns': []
        })

    # Replace "per cent" / "percent" with actual percentages from article
    percentage_count = 0
    for mapping in number_mappings:
        if '%' in mapping['original']:
            for pattern in mapping['patterns']:
                if re.search(pattern, summary, re.IGNORECASE):
                    summary = re.sub(pattern, mapping['original'], summary, count=1, flags=re.IGNORECASE)
                    percentage_count += 1
                    break

    return summary

def post_process_summary(summary, article, args=None):
    """PHASE 4: Post-process summary for better coverage and coherence."""
    import re

    # 1. Aggressive garbled text removal
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

    # 2. Reconstruct clean summary
    if clean_sentences:
        summary = '. '.join(clean_sentences)
        if not summary.endswith('.'):
            summary += '.'
    else:
        # Fallback: take first 100 words and clean them
        words = summary.split()[:100]
        clean_words = [w for w in words if len(w) <= 15 and re.search(r'[a-zA-Z]', w)]
        summary = ' '.join(clean_words[:50]) + '.'

    # 3. Restore numbers from article
    summary = restore_numbers_in_summary(summary, article)

    # 4. Final cleanup
    summary = re.sub(r'\s+', ' ', summary)  # Remove extra spaces
    summary = re.sub(r'\s*\.\s*', '. ', summary)  # Fix spacing around periods
    summary = summary.strip()

    return summary

def generate_summary(article, tokenizer, model, args=None):
    """Generate summary for the given article."""
    # Check if hybrid mode is enabled
    use_hybrid = args and getattr(args, 'hybrid', False)  # Default to quality over speed

    if use_hybrid and len(article.split()) > 300:
        # HYBRID MODE: Extract key sentences first
        extracted_content = extract_key_sentences(article, target_sentences=8)  # More sentences for better flow
        original_words = len(article.split())
        extracted_words = len(extracted_content.split())
        print(f"Hybrid mode: {original_words} -> {extracted_words} words ({extracted_words/original_words:.1%} compression)")

        # Add instruction for abstractive summarization
        abstraction_prompt = "Summarize the following key points into a coherent abstract summary: "
        cleaned_article = preprocess_text(abstraction_prompt + extracted_content)
    else:
        # STANDARD MODE: Use full article
        cleaned_article = preprocess_text(article)

    # PHASE 2: Entity-aware generation with explicit entity prompts
    if args and getattr(args, 'dynamic', False):
        # Extract key entities from the article
        entities = extract_key_entities(article)
        people = entities.get('PERSON', [])
        organizations = entities.get('ORG', [])

        # Extract key numbers from the article
        key_numbers = extract_key_numbers(article)

        # Create entity-aware prompt if entities found
        entity_prompt = ""
        if people or organizations or key_numbers:
            entity_list = []
            if people:
                entity_list.extend([f"people: {', '.join(people[:5])}"]) # Top 5 people
            if organizations:
                entity_list.extend([f"organizations: {', '.join(organizations[:5])}"]) # Top 5 orgs
            if key_numbers:
                entity_list.extend([f"key numbers: {', '.join(key_numbers[:5])}"]) # Top 5 numbers

            entity_prompt = f"Key information to include ({'; '.join(entity_list)}). "
            print(f"PHASE 2 ENHANCEMENT: Entity-aware prompting with {len(people)} people, {len(organizations)} organizations, {len(key_numbers)} key numbers")

        # PHASE 3: Enhanced abstractive instruction prompts
        if entity_prompt:
            # Create comprehensive instruction for abstractive summarization
            abstractive_instruction = (
                f"{entity_prompt}"
                "Write a coherent, well-structured summary that: "
                "1) Uses your own words to paraphrase key information, "
                "2) Connects ideas logically with proper transitions, "
                "3) Maintains factual accuracy while being concise, "
                "4) Includes the important entities and numbers mentioned above, "
                "5) Preserves all specific numbers, dates, percentages, and statistics. "
                "Summary: "
            )
            cleaned_article = abstractive_instruction + cleaned_article
            print(f"PHASE 3 ENHANCEMENT: Advanced abstractive instruction prompting applied")

    # Check if dynamic optimization is enabled
    if args and getattr(args, 'dynamic', False):
        # PHASE 1: Aggressive parameter optimization for enhanced abstractive output
        dynamic_params = calculate_dynamic_parameters(article)
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

        num_beams = dynamic_params['num_beams']
        # ENHANCED: Aggressive length penalty (2.5-3.0)
        length_penalty = max(2.5, dynamic_params['length_penalty'])

        print(f"PHASE 1 ENHANCEMENT: Aggressive parameter optimization active")
        print(f"Dynamic optimization: Article={dynamic_params['analysis']['article_length']} words")
        print(f"Complexity score: {dynamic_params['analysis']['complexity_score']}/7")
        print(f"Enhanced parameters: ratio={dynamic_params['ratio_used']:.2f}, beams={num_beams}, penalty={length_penalty}")
        print(f"Target summary: {max_tokens} words (min={min_length} - ENHANCED for better coverage)")

    elif args and getattr(args, 'adaptive_length', False):
        # Manual adaptive length (existing system)
        target_length, article_words = calculate_adaptive_length(article, args)
        max_tokens = target_length
        min_length = max(50, int(target_length * 0.6))
        num_beams = args.num_beams if args else 8
        length_penalty = args.length_penalty if args else 1.5
        print(f"Adaptive length: Article={article_words} words -> Target summary={target_length} words (min={min_length})")
    else:
        # Use provided args or defaults
        max_tokens = args.max_tokens if args else 512
        min_length = args.min_length if args else 100
        num_beams = args.num_beams if args else 8
        length_penalty = args.length_penalty if args else 1.5

    inputs = tokenizer(
        cleaned_article,
        max_length=4096,
        truncation=True,
        return_tensors="pt"
    )

    # PHASE 1 & 3: Enhanced generation with aggressive parameters for abstractive output
    summary_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        no_repeat_ngram_size=3,  # PHASE 3: Increased to reduce repetition and improve coherence
        length_penalty=length_penalty,  # Enhanced to 2.5-3.0 for aggressive length control
        early_stopping=True,
        max_new_tokens=max_tokens,
        min_length=min_length,  # Enhanced to 200+ words minimum
        do_sample=False,  # Keep deterministic for consistency
        repetition_penalty=1.5,  # ENHANCED: Stronger repetition penalty to force abstractive behavior
        forced_bos_token_id=None,  # Allow more natural generation
        pad_token_id=tokenizer.pad_token_id,
        encoder_no_repeat_ngram_size=2,  # PHASE 3: Prevent encoder repetitions
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # PHASE 4: Content coverage optimization and quality enhancement
    if args and getattr(args, 'dynamic', False):
        # Clean up garbled text and improve coherence
        summary = post_process_summary(summary, article, args)
        print(f"PHASE 4 ENHANCEMENT: Content coverage optimization applied")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Summarize any article using your RL checkpoint model")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Article text directly as argument")
    input_group.add_argument("--file", type=str, help="Path to text file containing article")
    input_group.add_argument("--stdin", action="store_true", help="Read article from stdin")

    # Optional arguments
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_DIR,
                       help=f"Path to checkpoint directory (default: {CHECKPOINT_DIR})")
    parser.add_argument("--compare", type=str, help="Path to comparison model checkpoint for side-by-side evaluation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output the summary")

    # Mode selection (simplified interface)
    parser.add_argument("--mode", type=str, choices=["basic", "smart", "advanced"], default="smart",
                       help="Analysis mode: basic (just summary), smart (auto-optimize + classify + evaluate), advanced (manual control)")

    # Generation parameters (for advanced mode)
    parser.add_argument("--num_beams", type=int, default=8, help="Number of beams for beam search (advanced mode)")
    parser.add_argument("--length_penalty", type=float, default=1.5, help="Length penalty for generation (advanced mode)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum new tokens to generate (advanced mode)")
    parser.add_argument("--min_length", type=int, default=100, help="Minimum length for summary (advanced mode)")
    parser.add_argument("--evaluate", action="store_true", help="Show detailed evaluation metrics (advanced mode)")

    # Adaptive length parameters (for advanced mode)
    parser.add_argument("--adaptive_length", action="store_true", help="Enable adaptive summary length based on article length (advanced mode)")
    parser.add_argument("--summary_ratio", type=float, default=0.15, help="Target summary length as ratio of article length (advanced mode)")
    parser.add_argument("--adaptive_max", type=int, default=1024, help="Maximum summary length for adaptive mode (advanced mode)")
    parser.add_argument("--adaptive_min", type=int, default=50, help="Minimum summary length for adaptive mode (advanced mode)")

    # Dynamic optimization (for advanced mode)
    parser.add_argument("--dynamic", action="store_true", help="Enable fully dynamic parameter optimization (advanced mode)")

    # Article classification (for advanced mode)
    parser.add_argument("--classify", action="store_true", help="Enable article topic classification (advanced mode)")
    parser.add_argument("--classification_categories", type=str, help="Comma-separated custom categories for classification (advanced mode)")

    # Simple toggles
    parser.add_argument("--no_classify", action="store_true", help="Disable classification in smart mode")
    parser.add_argument("--no_evaluate", action="store_true", help="Disable evaluation metrics in smart mode")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid extractive-abstractive mode for speed")
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid extractive-abstractive mode")
    parser.add_argument("--ultra_fast", action="store_true", help="Ultra-fast mode with aggressive optimizations")

    args = parser.parse_args()

    # Apply mode-based defaults
    if args.mode == "smart":
        # Smart mode: enable best features automatically
        if not args.no_classify:
            args.classify = True
        if not args.no_evaluate:
            args.evaluate = True
        args.dynamic = True
        args.hybrid = False  # Default to quality over speed
    elif args.mode == "basic":
        # Basic mode: minimal output
        args.classify = False
        args.evaluate = False
        args.dynamic = False
        args.hybrid = False
    # Advanced mode uses whatever user specified
    if not hasattr(args, 'hybrid'):
        args.hybrid = not args.no_hybrid

    # Get article text based on input method
    if args.text:
        article = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                article = f.read().strip()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.stdin:
        try:
            article = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nOperation cancelled.", file=sys.stderr)
            sys.exit(1)

    if not article:
        print("Error: No article text provided or file is empty.", file=sys.stderr)
        sys.exit(1)

    # Perform article classification if requested
    if args.classify:
        try:
            import subprocess
            import json

            # Prepare classification command
            classify_cmd = [sys.executable, "classify_article.py", "--text", article, "--quiet"]

            if args.classification_categories:
                classify_cmd.extend(["--categories", args.classification_categories])

            # Run classification
            result = subprocess.run(classify_cmd, capture_output=True, text=True, cwd=".")

            if result.returncode == 0:
                article_category = result.stdout.strip()
            else:
                print(f"Classification warning: {result.stderr.strip()}")
                article_category = "unknown"
        except Exception as e:
            print(f"Classification error: {e}")
            article_category = "unknown"

    try:
        # Load main model
        tokenizer, model = load_model(args.checkpoint)

        # Load comparison model if specified
        compare_tokenizer, compare_model = None, None
        if args.compare:
            print(f"\nLoading comparison model from: {args.compare}")
            compare_tokenizer, compare_model = load_model(args.compare)

        # Display article
        if not args.quiet:
            print("\n" + "="*80)
            print("ARTICLE (PREPROCESSED):")
            print(preprocess_text(article))
            print("\n" + "="*80)

        # Generate summary from main model
        summary = generate_summary(article, tokenizer, model, args)

        if args.compare and compare_model:
            # Generate summary from comparison model
            compare_summary = generate_summary(article, compare_tokenizer, compare_model, args)

            if not args.quiet:
                # Show classification first if enabled
                if args.classify:
                    print(f"ARTICLE CATEGORY: {article_category.upper()}")
                    print("="*80)

                print("MAIN MODEL SUMMARY:")
                print(summary)
                print("\n" + "-"*80)
                print("COMPARISON MODEL SUMMARY:")
                print(compare_summary)
                print("="*80)
            else:
                print("Main:", summary)
                print("Compare:", compare_summary)

            # Show evaluation if requested
            if args.evaluate:
                print("\n" + "="*80)
                print("EVALUATION METRICS:")
                print("="*80)
                coverage_main = calculate_coverage_score(article, summary)
                coverage_compare = calculate_coverage_score(article, compare_summary)
                print(f"Entity Coverage - Main: {coverage_main:.2%}, Compare: {coverage_compare:.2%}")
                print(f"Length - Main: {len(summary.split())} words, Compare: {len(compare_summary.split())} words")
        else:
            if not args.quiet:
                # Show classification first if enabled
                if args.classify:
                    print(f"ARTICLE CATEGORY: {article_category.upper()}")
                    print("="*80)

                print("GENERATED SUMMARY:")
            print(summary)
            if not args.quiet:
                print("="*80)

            # Show evaluation if requested
            if args.evaluate:
                print("\n" + "="*80)
                print("EVALUATION METRICS:")
                print("="*80)
                coverage = calculate_coverage_score(article, summary)
                print(f"Entity Coverage: {coverage:.2%}")
                print(f"Summary Length: {len(summary.split())} words")

                # Show key entities
                article_entities = extract_key_entities(article)
                summary_entities = extract_key_entities(summary)
                print(f"\nKey Entities in Article: {len(article_entities['PERSON']) + len(article_entities['ORG'])}")
                print(f"People: {', '.join(article_entities['PERSON'][:5])}")  # Show first 5
                print(f"Organizations: {', '.join(article_entities['ORG'][:5])}")
                print(f"\nEntities in Summary: {len(summary_entities['PERSON']) + len(summary_entities['ORG'])}")
                print("="*80)

    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()