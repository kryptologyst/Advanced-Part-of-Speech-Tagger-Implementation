# Project 164: Advanced Part-of-Speech Tagger Implementation
# ==========================================================
# 
# This is the original simple implementation. For the full-featured version,
# see pos_tagger.py which includes:
# - Multiple model support (SpaCy, NLTK, Transformers)
# - Confidence scoring and uncertainty quantification
# - Batch processing capabilities
# - Mock database integration
# - Modern web interface (app.py)
# - Comprehensive testing suite
# - Advanced visualizations
#
# To run the full version:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Download SpaCy model: python -m spacy download en_core_web_sm
# 3. Run web interface: streamlit run app.py
# 4. Or run demo: python demo.py
#
# Original Implementation: POS Tagging with SpaCy
# ================================================

import spacy

def main():
    """Original simple POS tagging implementation"""
    print("🧠 Part-of-Speech Tagging (Original Implementation)")
    print("=" * 60)
    
    try:
        # Load the English SpaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Example sentence or paragraph
        text = "The quick brown fox jumps over the lazy dog near the riverbank."
        
        # Process the text
        doc = nlp(text)
        
        # Print each token with its POS tag and detailed tag
        print(f"Text: {text}")
        print(f"\n{'Token':<15} {'POS':<10} {'Tag':<10} {'Explanation'}")
        print("-" * 60)
        
        for token in doc:
            explanation = spacy.explain(token.tag_) or "Unknown tag"
            print(f"{token.text:<15} {token.pos_:<10} {token.tag_:<10} {explanation}")
        
        print(f"\n✅ Processed {len(doc)} tokens successfully!")
        
    except OSError:
        print("❌ SpaCy English model not found!")
        print("Please run: python -m spacy download en_core_web_sm")
        print("\nOr use the advanced implementation: python pos_tagger.py")
    
    print("\n🧠 What This Project Demonstrates:")
    print("• Applies SpaCy's POS tagger to annotate text with part-of-speech information")
    print("• Differentiates between universal POS tags and detailed grammar-specific tags")
    print("• Explains each tag in human-readable language")
    print("\n📖 For advanced features, see pos_tagger.py and app.py")

if __name__ == "__main__":
    main()