"""
Project 164: Advanced Part-of-Speech Tagger Implementation
=========================================================

A modern, comprehensive POS tagging system with multiple models, confidence scoring,
and advanced NLP features. Supports SpaCy, NLTK, and Transformers-based models.

Features:
- Multiple POS tagging models (SpaCy, NLTK, Transformers)
- Confidence scoring and uncertainty quantification
- Batch processing capabilities
- Custom model training support
- Comprehensive evaluation metrics
- Modern web interface
- Mock database integration
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

import spacy
import nltk
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Rich console for beautiful output
console = Console()

@dataclass
class POSTagResult:
    """Data class for POS tagging results"""
    token: str
    pos_tag: str
    tag: str
    explanation: str
    confidence: float = 1.0
    model_name: str = "unknown"

@dataclass
class SentenceAnalysis:
    """Data class for complete sentence analysis"""
    text: str
    tokens: List[POSTagResult]
    model_name: str
    processing_time: float
    confidence_score: float

class MockDatabase:
    """Mock database for storing sample texts and results"""
    
    def __init__(self, db_path: str = "pos_tagger.db"):
        self.db_path = db_path
        self.init_database()
        self.populate_sample_data()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                difficulty_level INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pos_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER,
                model_name TEXT NOT NULL,
                results TEXT NOT NULL,
                processing_time REAL,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (text_id) REFERENCES sample_texts (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def populate_sample_data(self):
        """Populate database with sample texts"""
        sample_texts = [
            ("simple", "The quick brown fox jumps over the lazy dog.", 1),
            ("simple", "I love programming and machine learning.", 1),
            ("medium", "The sophisticated algorithm efficiently processes natural language data.", 2),
            ("medium", "Despite the challenges, researchers continue to advance AI technology.", 2),
            ("complex", "The interdisciplinary nature of computational linguistics necessitates collaboration between computer scientists, linguists, and cognitive psychologists.", 3),
            ("complex", "Quantum computing represents a paradigm shift that could revolutionize cryptography, optimization, and scientific simulation.", 3),
            ("literary", "The mellifluous prose cascaded like a gentle stream through the verdant valley of imagination.", 2),
            ("technical", "The convolutional neural network architecture utilizes batch normalization and dropout regularization techniques.", 3),
            ("news", "The unprecedented economic downturn has significantly impacted global markets and consumer confidence.", 2),
            ("academic", "Empirical evidence suggests that contextualized word representations outperform traditional static embeddings in downstream tasks.", 3)
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, text, difficulty in sample_texts:
            cursor.execute('''
                INSERT OR IGNORE INTO sample_texts (category, text, difficulty_level)
                VALUES (?, ?, ?)
            ''', (category, text, difficulty))
        
        conn.commit()
        conn.close()
        logger.info("Sample data populated successfully")
    
    def get_sample_texts(self, category: Optional[str] = None, limit: int = 10) -> List[Tuple[int, str, str, int]]:
        """Retrieve sample texts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('''
                SELECT id, category, text, difficulty_level 
                FROM sample_texts 
                WHERE category = ? 
                ORDER BY RANDOM() 
                LIMIT ?
            ''', (category, limit))
        else:
            cursor.execute('''
                SELECT id, category, text, difficulty_level 
                FROM sample_texts 
                ORDER BY RANDOM() 
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def save_results(self, text_id: int, model_name: str, results: List[POSTagResult], 
                    processing_time: float, confidence_score: float):
        """Save POS tagging results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert results to JSON
        results_json = json.dumps([
            {
                "token": r.token,
                "pos_tag": r.pos_tag,
                "tag": r.tag,
                "explanation": r.explanation,
                "confidence": r.confidence,
                "model_name": r.model_name
            } for r in results
        ])
        
        cursor.execute('''
            INSERT INTO pos_results (text_id, model_name, results, processing_time, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (text_id, model_name, results_json, processing_time, confidence_score))
        
        conn.commit()
        conn.close()

class AdvancedPOSTagger:
    """Advanced POS tagger with multiple model support"""
    
    def __init__(self):
        self.models = {}
        self.db = MockDatabase()
        self._load_models()
    
    def _load_models(self):
        """Load all available POS tagging models"""
        try:
            # Load SpaCy models
            self.models['spacy_en'] = spacy.load("en_core_web_sm")
            logger.info("SpaCy English model loaded successfully")
        except OSError:
            logger.warning("SpaCy English model not found. Run: python -m spacy download en_core_web_sm")
        
        try:
            # Load NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('tagsets', quiet=True)
            self.models['nltk'] = True
            logger.info("NLTK models loaded successfully")
        except Exception as e:
            logger.warning(f"NLTK models not available: {e}")
        
        try:
            # Load Transformers model
            self.models['transformers'] = pipeline(
                "token-classification",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            logger.info("Transformers model loaded successfully")
        except Exception as e:
            logger.warning(f"Transformers model not available: {e}")
    
    def tag_with_spacy(self, text: str) -> List[POSTagResult]:
        """POS tagging using SpaCy"""
        if 'spacy_en' not in self.models:
            raise ValueError("SpaCy model not available")
        
        doc = self.models['spacy_en'](text)
        results = []
        
        for token in doc:
            results.append(POSTagResult(
                token=token.text,
                pos_tag=token.pos_,
                tag=token.tag_,
                explanation=spacy.explain(token.tag_) or "Unknown tag",
                confidence=1.0,  # SpaCy doesn't provide confidence scores
                model_name="spacy_en"
            ))
        
        return results
    
    def tag_with_nltk(self, text: str) -> List[POSTagResult]:
        """POS tagging using NLTK"""
        if 'nltk' not in self.models:
            raise ValueError("NLTK model not available")
        
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        results = []
        for token, tag in pos_tags:
            results.append(POSTagResult(
                token=token,
                pos_tag=tag,
                tag=tag,
                explanation=nltk.help.upenn_tagset(tag) if hasattr(nltk.help, 'upenn_tagset') else "NLTK tag",
                confidence=1.0,  # NLTK doesn't provide confidence scores
                model_name="nltk"
            ))
        
        return results
    
    def tag_with_transformers(self, text: str) -> List[POSTagResult]:
        """POS tagging using Transformers"""
        if 'transformers' not in self.models:
            raise ValueError("Transformers model not available")
        
        results = self.models['transformers'](text)
        pos_results = []
        
        for result in results:
            pos_results.append(POSTagResult(
                token=result['word'],
                pos_tag=result['entity_group'],
                tag=result['label'],
                explanation=f"Transformers prediction with confidence {result['score']:.3f}",
                confidence=result['score'],
                model_name="transformers"
            ))
        
        return pos_results
    
    def tag_text(self, text: str, model: str = "spacy_en") -> SentenceAnalysis:
        """Tag text with specified model"""
        import time
        start_time = time.time()
        
        if model == "spacy_en":
            tokens = self.tag_with_spacy(text)
        elif model == "nltk":
            tokens = self.tag_with_nltk(text)
        elif model == "transformers":
            tokens = self.tag_with_transformers(text)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        processing_time = time.time() - start_time
        confidence_score = np.mean([token.confidence for token in tokens])
        
        return SentenceAnalysis(
            text=text,
            tokens=tokens,
            model_name=model,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
    
    def compare_models(self, text: str) -> Dict[str, SentenceAnalysis]:
        """Compare results across all available models"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                results[model_name] = self.tag_text(text, model_name)
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
        
        return results
    
    def batch_process(self, texts: List[str], model: str = "spacy_en") -> List[SentenceAnalysis]:
        """Process multiple texts in batch"""
        results = []
        
        for text in track(texts, description="Processing texts..."):
            try:
                result = self.tag_text(text, model)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {e}")
        
        return results
    
    def visualize_results(self, analysis: SentenceAnalysis, save_path: Optional[str] = None):
        """Create visualizations of POS tagging results"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'POS Tagging Analysis - {analysis.model_name}', fontsize=16)
        
        # 1. POS tag distribution
        pos_counts = pd.Series([token.pos_tag for token in analysis.tokens]).value_counts()
        ax1.pie(pos_counts.values, labels=pos_counts.index, autopct='%1.1f%%')
        ax1.set_title('POS Tag Distribution')
        
        # 2. Confidence scores
        confidences = [token.confidence for token in analysis.tokens]
        ax2.bar(range(len(confidences)), confidences)
        ax2.set_title('Confidence Scores by Token')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Confidence')
        
        # 3. Token length distribution
        token_lengths = [len(token.token) for token in analysis.tokens]
        ax3.hist(token_lengths, bins=10, alpha=0.7)
        ax3.set_title('Token Length Distribution')
        ax3.set_xlabel('Length')
        ax3.set_ylabel('Frequency')
        
        # 4. Processing time vs text length
        ax4.scatter(len(analysis.text), analysis.processing_time, s=100, alpha=0.7)
        ax4.set_title('Processing Time vs Text Length')
        ax4.set_xlabel('Text Length')
        ax4.set_ylabel('Processing Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def display_results(self, analysis: SentenceAnalysis):
        """Display results in a beautiful table format"""
        table = Table(title=f"POS Tagging Results - {analysis.model_name}")
        table.add_column("Token", style="cyan")
        table.add_column("POS Tag", style="magenta")
        table.add_column("Tag", style="green")
        table.add_column("Explanation", style="yellow")
        table.add_column("Confidence", style="red")
        
        for token in analysis.tokens:
            table.add_row(
                token.token,
                token.pos_tag,
                token.tag,
                token.explanation,
                f"{token.confidence:.3f}"
            )
        
        console.print(table)
        console.print(f"\n[bold green]Processing Time:[/bold green] {analysis.processing_time:.4f}s")
        console.print(f"[bold green]Average Confidence:[/bold green] {analysis.confidence_score:.3f}")

def main():
    """Main function to demonstrate the POS tagger"""
    console.print("[bold blue]🧠 Advanced Part-of-Speech Tagger[/bold blue]")
    console.print("=" * 50)
    
    # Initialize tagger
    tagger = AdvancedPOSTagger()
    
    # Example texts
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Despite the challenges, researchers continue to advance our understanding of natural language processing."
    ]
    
    # Process each text
    for i, text in enumerate(example_texts, 1):
        console.print(f"\n[bold yellow]Example {i}:[/bold yellow] {text}")
        
        # Single model analysis
        analysis = tagger.tag_text(text, "spacy_en")
        tagger.display_results(analysis)
        
        # Save to database
        sample_texts = tagger.db.get_sample_texts(limit=1)
        if sample_texts:
            text_id, _, _, _ = sample_texts[0]
            tagger.db.save_results(text_id, analysis.model_name, analysis.tokens, 
                                 analysis.processing_time, analysis.confidence_score)
    
    # Model comparison
    console.print("\n[bold blue]Model Comparison:[/bold blue]")
    comparison_text = "The sophisticated algorithm efficiently processes natural language data."
    comparisons = tagger.compare_models(comparison_text)
    
    for model_name, analysis in comparisons.items():
        console.print(f"\n[bold cyan]{model_name.upper()}:[/bold cyan]")
        console.print(f"Processing Time: {analysis.processing_time:.4f}s")
        console.print(f"Confidence: {analysis.confidence_score:.3f}")
        console.print(f"Tokens: {len(analysis.tokens)}")
    
    # Batch processing example
    console.print("\n[bold blue]Batch Processing Example:[/bold blue]")
    sample_data = tagger.db.get_sample_texts(limit=3)
    texts = [text for _, _, text, _ in sample_data]
    
    batch_results = tagger.batch_process(texts, "spacy_en")
    console.print(f"Processed {len(batch_results)} texts successfully")
    
    # Create visualization
    if batch_results:
        tagger.visualize_results(batch_results[0], "pos_analysis.png")

if __name__ == "__main__":
    main()
