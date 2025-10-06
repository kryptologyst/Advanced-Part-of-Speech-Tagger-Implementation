#!/usr/bin/env python3
"""
Demo script for Advanced POS Tagger
===================================

A simple demonstration of the POS tagger capabilities.
"""

from pos_tagger import AdvancedPOSTagger
import time

def main():
    """Run the demo"""
    print("🧠 Advanced Part-of-Speech Tagger Demo")
    print("=" * 50)
    
    # Initialize tagger
    print("🔄 Initializing POS tagger...")
    tagger = AdvancedPOSTagger()
    print("✅ Tagger initialized successfully!")
    
    # Demo texts
    demo_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Despite the challenges, researchers continue to advance our understanding of natural language processing.",
        "The sophisticated algorithm efficiently processes complex linguistic patterns.",
        "Quantum computing represents a paradigm shift in computational capabilities."
    ]
    
    print(f"\n📝 Processing {len(demo_texts)} demo texts...")
    
    # Process each text
    for i, text in enumerate(demo_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Text: {text}")
        
        # Analyze with SpaCy
        start_time = time.time()
        analysis = tagger.tag_text(text, "spacy_en")
        processing_time = time.time() - start_time
        
        print(f"Model: {analysis.model_name}")
        print(f"Processing time: {processing_time:.4f}s")
        print(f"Confidence: {analysis.confidence_score:.3f}")
        print(f"Tokens: {len(analysis.tokens)}")
        
        # Show first few tokens
        print("First 5 tokens:")
        for j, token in enumerate(analysis.tokens[:5]):
            print(f"  {token.token} -> {token.pos_tag} ({token.tag})")
        
        if len(analysis.tokens) > 5:
            print(f"  ... and {len(analysis.tokens) - 5} more tokens")
    
    # Model comparison
    print(f"\n🔄 Comparing models on sample text...")
    comparison_text = "The sophisticated algorithm efficiently processes natural language data."
    comparisons = tagger.compare_models(comparison_text)
    
    print(f"\n📊 Model Comparison Results:")
    print(f"Text: {comparison_text}")
    print("-" * 60)
    
    for model_name, analysis in comparisons.items():
        print(f"{model_name.upper():<15} | Time: {analysis.processing_time:.4f}s | Confidence: {analysis.confidence_score:.3f} | Tokens: {len(analysis.tokens)}")
    
    # Batch processing demo
    print(f"\n🔄 Batch processing demo...")
    batch_texts = demo_texts[:3]  # Use first 3 texts
    batch_results = tagger.batch_process(batch_texts, "spacy_en")
    
    print(f"✅ Processed {len(batch_results)} texts in batch")
    total_time = sum(r.processing_time for r in batch_results)
    avg_confidence = sum(r.confidence_score for r in batch_results) / len(batch_results)
    
    print(f"Total processing time: {total_time:.4f}s")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # Database demo
    print(f"\n🗄️ Database demo...")
    sample_texts = tagger.db.get_sample_texts(limit=3)
    print(f"Retrieved {len(sample_texts)} sample texts from database:")
    
    for text_id, category, text, difficulty in sample_texts:
        print(f"  [{category}] (Level {difficulty}): {text[:50]}...")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\n📖 Next steps:")
    print(f"1. Run the web interface: streamlit run app.py")
    print(f"2. Run tests: python -m pytest test_pos_tagger.py -v")
    print(f"3. Explore the code in pos_tagger.py")

if __name__ == "__main__":
    main()
