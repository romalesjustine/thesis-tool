#!/usr/bin/env python
"""
classify_article.py

Zero-shot article classification script that automatically categorizes news articles
into predefined topics like political, sports, environmental, technology, etc.

Usage examples:
  python classify_article.py --text "Your article text here..."
  python classify_article.py --file article.txt
  python classify_article.py --file article.txt --custom_categories "technology,science,health"
"""

import argparse
import sys
from transformers import pipeline

# Fast core categories - optimized for speed
DEFAULT_CATEGORIES = [
    "political",
    "sports",
    "business",
    "technology",
    "health",
    "science",
    "environmental",
    "entertainment",
    "crime"
]

# Extended categories for fallback
EXTENDED_CATEGORIES = [
    "government", "election", "policy",
    "athletics", "competition", "games",
    "economy", "finance", "market", "corporate",
    "ai", "digital", "innovation", "software",
    "medicine", "medical", "healthcare", "clinical",
    "scientific", "study", "discovery", "laboratory",
    "climate", "sustainability", "pollution", "nature",
    "celebrity", "media", "culture", "music",
    "academic", "university", "learning", "student",
    "global", "foreign", "diplomatic", "worldwide",
    "legal", "court", "justice", "investigation",
    "community", "society", "public", "cultural",
    "animals", "biodiversity", "habitat", "species",
    "analysis", "findings", "data", "methodology",
    "protection", "preserve", "ecosystem", "sanctuary"
]

# Global classifier to avoid reloading
CLASSIFIER = None

def load_classifier(quiet=False):
    """Load the zero-shot classification model."""
    global CLASSIFIER
    if CLASSIFIER is None:
        if not quiet:
            print("Loading zero-shot classification model...")
        try:
            CLASSIFIER = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU
            )
            if not quiet:
                print("Classification model loaded successfully!")
        except Exception as e:
            if not quiet:
                print(f"Error loading classification model: {e}")
                print("Falling back to text-based classification...")
            CLASSIFIER = "fallback"
    return CLASSIFIER

def fallback_classify(text, categories):
    """Simple keyword-based classification fallback."""
    text_lower = text.lower()
    scores = {}

    # Comprehensive keywords for each category
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

    for category in categories:
        if category.lower() in keywords:
            keyword_list = keywords[category.lower()]
            score = sum(1 for keyword in keyword_list if keyword in text_lower)
            scores[category] = score / len(keyword_list)  # Normalize by number of keywords
        else:
            scores[category] = 0

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Convert to expected format
    result = {
        'labels': [item[0] for item in sorted_scores],
        'scores': [item[1] for item in sorted_scores]
    }

    return result

def classify_article_fast(text, categories=None, threshold=0.05, quiet=False):
    """Fast classification - keyword first, then AI if needed."""
    if categories is None:
        categories = DEFAULT_CATEGORIES

    # Try fast keyword matching first
    keyword_result = fallback_classify(text, categories)
    if keyword_result['scores'][0] >= 0.1:  # Good keyword match
        return [(keyword_result['labels'][0], keyword_result['scores'][0])]

    # Only use AI model if keyword matching is weak
    classifier = load_classifier(quiet=quiet)

    if classifier == "fallback":
        return [(keyword_result['labels'][0], keyword_result['scores'][0])]

    try:
        # Truncate text for speed
        max_length = 500  # Reduced from 1000 for speed
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])

        result = classifier(text, categories)

        # Take first result above threshold
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                return [(label, score)]

        # Fallback to keyword result
        return [(keyword_result['labels'][0], max(keyword_result['scores'][0], 0.05))]

    except Exception as e:
        return [(keyword_result['labels'][0], keyword_result['scores'][0])]

def classify_article(text, categories=None, threshold=0.05):
    """Wrapper for backward compatibility."""
    return classify_article_fast(text, categories, threshold)

def analyze_article_topic(text, categories=None, show_all=False):
    """Analyze and display article topic classification."""
    print("ARTICLE TOPIC ANALYSIS")
    print("=" * 50)

    results = classify_article(text, categories)

    if not results:
        print("No clear category detected (all scores below threshold)")
        return None

    # Show primary classification
    primary_category, primary_score = results[0]
    print(f"PRIMARY TOPIC: {primary_category.upper()}")
    print(f"Confidence: {primary_score:.1%}")

    if show_all and len(results) > 1:
        print(f"\nALL CLASSIFICATIONS:")
        for i, (category, score) in enumerate(results[:5], 1):
            print(f"{i}. {category}: {score:.1%}")

    print("=" * 50)
    return primary_category

def main():
    parser = argparse.ArgumentParser(description="Classify news articles using zero-shot classification")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Article text directly as argument")
    input_group.add_argument("--file", type=str, help="Path to text file containing article")
    input_group.add_argument("--stdin", action="store_true", help="Read article from stdin")

    # Classification options
    parser.add_argument("--categories", type=str, help="Comma-separated custom categories (e.g. 'technology,science,health')")
    parser.add_argument("--threshold", type=float, default=0.1, help="Minimum confidence threshold (default: 0.1)")
    parser.add_argument("--show_all", action="store_true", help="Show all classification scores")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output the primary category")

    args = parser.parse_args()

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

    # Parse custom categories if provided
    categories = None
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
        print(f"Using custom categories: {', '.join(categories)}")

    try:
        if args.quiet:
            # Only show primary category
            results = classify_article_fast(article, categories, args.threshold, quiet=True)
            if results:
                print(results[0][0])
            else:
                print("unknown")
        else:
            # Full analysis
            analyze_article_topic(article, categories, args.show_all)

    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()