"""
Comprehensive Test Suite for Advanced POS Tagger
===============================================

Tests for all components including models, database, and web interface.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sqlite3
import json
from typing import List, Dict

# Import our modules
from pos_tagger import (
    AdvancedPOSTagger, 
    MockDatabase, 
    POSTagResult, 
    SentenceAnalysis
)

class TestPOSTagResult(unittest.TestCase):
    """Test POSTagResult dataclass"""
    
    def test_postag_result_creation(self):
        """Test POSTagResult creation with all parameters"""
        result = POSTagResult(
            token="test",
            pos_tag="NOUN",
            tag="NN",
            explanation="Noun",
            confidence=0.95,
            model_name="test_model"
        )
        
        self.assertEqual(result.token, "test")
        self.assertEqual(result.pos_tag, "NOUN")
        self.assertEqual(result.tag, "NN")
        self.assertEqual(result.explanation, "Noun")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.model_name, "test_model")
    
    def test_postag_result_defaults(self):
        """Test POSTagResult creation with default values"""
        result = POSTagResult(
            token="test",
            pos_tag="NOUN",
            tag="NN",
            explanation="Noun"
        )
        
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.model_name, "unknown")

class TestSentenceAnalysis(unittest.TestCase):
    """Test SentenceAnalysis dataclass"""
    
    def test_sentence_analysis_creation(self):
        """Test SentenceAnalysis creation"""
        tokens = [
            POSTagResult("test", "NOUN", "NN", "Noun"),
            POSTagResult("word", "NOUN", "NN", "Noun")
        ]
        
        analysis = SentenceAnalysis(
            text="test word",
            tokens=tokens,
            model_name="test_model",
            processing_time=0.1,
            confidence_score=0.95
        )
        
        self.assertEqual(analysis.text, "test word")
        self.assertEqual(len(analysis.tokens), 2)
        self.assertEqual(analysis.model_name, "test_model")
        self.assertEqual(analysis.processing_time, 0.1)
        self.assertEqual(analysis.confidence_score, 0.95)

class TestMockDatabase(unittest.TestCase):
    """Test MockDatabase functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = MockDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('sample_texts', tables)
        self.assertIn('pos_results', tables)
        
        conn.close()
    
    def test_sample_data_population(self):
        """Test sample data population"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sample_texts")
        count = cursor.fetchone()[0]
        
        self.assertGreater(count, 0)
        
        conn.close()
    
    def test_get_sample_texts(self):
        """Test retrieving sample texts"""
        texts = self.db.get_sample_texts(limit=5)
        
        self.assertIsInstance(texts, list)
        self.assertLessEqual(len(texts), 5)
        
        if texts:
            text_id, category, text, difficulty = texts[0]
            self.assertIsInstance(text_id, int)
            self.assertIsInstance(category, str)
            self.assertIsInstance(text, str)
            self.assertIsInstance(difficulty, int)
    
    def test_get_sample_texts_by_category(self):
        """Test retrieving sample texts by category"""
        texts = self.db.get_sample_texts(category="simple", limit=3)
        
        self.assertIsInstance(texts, list)
        
        for text_id, category, text, difficulty in texts:
            self.assertEqual(category, "simple")
    
    def test_save_results(self):
        """Test saving POS results"""
        # Get a sample text
        sample_texts = self.db.get_sample_texts(limit=1)
        if sample_texts:
            text_id, _, _, _ = sample_texts[0]
            
            # Create test results
            results = [
                POSTagResult("test", "NOUN", "NN", "Noun", 0.95, "test_model"),
                POSTagResult("word", "NOUN", "NN", "Noun", 0.90, "test_model")
            ]
            
            # Save results
            self.db.save_results(text_id, "test_model", results, 0.1, 0.925)
            
            # Verify results were saved
            conn = sqlite3.connect(self.temp_db.name)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM pos_results WHERE text_id = ?", (text_id,))
            count = cursor.fetchone()[0]
            
            self.assertEqual(count, 1)
            
            conn.close()

class TestAdvancedPOSTagger(unittest.TestCase):
    """Test AdvancedPOSTagger functionality"""
    
    def setUp(self):
        """Set up test tagger"""
        # Mock the models to avoid loading actual models in tests
        self.tagger = AdvancedPOSTagger()
        self.tagger.models = {}  # Clear models for testing
    
    @patch('spacy.load')
    def test_load_spacy_model(self, mock_spacy_load):
        """Test loading SpaCy model"""
        mock_model = Mock()
        mock_spacy_load.return_value = mock_model
        
        self.tagger._load_models()
        
        if 'spacy_en' in self.tagger.models:
            mock_spacy_load.assert_called_with("en_core_web_sm")
    
    @patch('nltk.download')
    @patch('nltk.word_tokenize')
    @patch('nltk.pos_tag')
    def test_nltk_functionality(self, mock_pos_tag, mock_word_tokenize, mock_download):
        """Test NLTK functionality"""
        mock_word_tokenize.return_value = ["test", "word"]
        mock_pos_tag.return_value = [("test", "NN"), ("word", "NN")]
        
        self.tagger.models['nltk'] = True
        
        results = self.tagger.tag_with_nltk("test word")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].token, "test")
        self.assertEqual(results[0].pos_tag, "NN")
        self.assertEqual(results[0].model_name, "nltk")
    
    def test_tag_with_spacy_no_model(self):
        """Test SpaCy tagging when model not available"""
        with self.assertRaises(ValueError):
            self.tagger.tag_with_spacy("test text")
    
    def test_tag_with_nltk_no_model(self):
        """Test NLTK tagging when model not available"""
        with self.assertRaises(ValueError):
            self.tagger.tag_with_nltk("test text")
    
    def test_tag_with_transformers_no_model(self):
        """Test Transformers tagging when model not available"""
        with self.assertRaises(ValueError):
            self.tagger.tag_with_transformers("test text")
    
    def test_tag_text_unknown_model(self):
        """Test tagging with unknown model"""
        with self.assertRaises(ValueError):
            self.tagger.tag_text("test text", "unknown_model")
    
    def test_batch_process_empty_list(self):
        """Test batch processing with empty list"""
        results = self.tagger.batch_process([])
        self.assertEqual(len(results), 0)
    
    def test_compare_models_no_models(self):
        """Test model comparison with no models available"""
        comparisons = self.tagger.compare_models("test text")
        self.assertEqual(len(comparisons), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
    
    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    @patch('spacy.load')
    def test_full_workflow(self, mock_spacy_load):
        """Test complete workflow"""
        # Mock SpaCy model
        mock_model = Mock()
        mock_doc = Mock()
        mock_token1 = Mock()
        mock_token1.text = "test"
        mock_token1.pos_ = "NOUN"
        mock_token1.tag_ = "NN"
        mock_token2 = Mock()
        mock_token2.text = "word"
        mock_token2.pos_ = "NOUN"
        mock_token2.tag_ = "NN"
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        mock_model.return_value = mock_doc
        mock_spacy_load.return_value = mock_model
        
        # Mock spacy.explain
        with patch('spacy.explain', return_value="Noun"):
            # Initialize tagger
            tagger = AdvancedPOSTagger()
            tagger.db = MockDatabase(self.temp_db.name)
            
            # Test text
            test_text = "test word"
            
            # Tag text
            analysis = tagger.tag_text(test_text, "spacy_en")
            
            # Verify results
            self.assertEqual(analysis.text, test_text)
            self.assertEqual(len(analysis.tokens), 2)
            self.assertEqual(analysis.model_name, "spacy_en")
            self.assertGreater(analysis.processing_time, 0)
            self.assertGreater(analysis.confidence_score, 0)

# Pytest fixtures and tests
@pytest.fixture
def temp_database():
    """Create temporary database for testing"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db = MockDatabase(temp_db.name)
    yield db
    os.unlink(temp_db.name)

@pytest.fixture
def mock_tagger():
    """Create mock tagger for testing"""
    tagger = AdvancedPOSTagger()
    tagger.models = {}  # Clear models
    return tagger

def test_database_operations(temp_database):
    """Test database operations with pytest"""
    # Test getting sample texts
    texts = temp_database.get_sample_texts(limit=3)
    assert len(texts) <= 3
    
    # Test saving results
    if texts:
        text_id, _, _, _ = texts[0]
        results = [
            POSTagResult("test", "NOUN", "NN", "Noun", 0.95, "test_model")
        ]
        
        temp_database.save_results(text_id, "test_model", results, 0.1, 0.95)
        
        # Verify save
        conn = sqlite3.connect(temp_database.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pos_results WHERE text_id = ?", (text_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        conn.close()

def test_postag_result_serialization():
    """Test POSTagResult serialization"""
    result = POSTagResult("test", "NOUN", "NN", "Noun", 0.95, "test_model")
    
    # Convert to dict
    result_dict = {
        "token": result.token,
        "pos_tag": result.pos_tag,
        "tag": result.tag,
        "explanation": result.explanation,
        "confidence": result.confidence,
        "model_name": result.model_name
    }
    
    # Test JSON serialization
    json_str = json.dumps(result_dict)
    parsed_dict = json.loads(json_str)
    
    assert parsed_dict["token"] == "test"
    assert parsed_dict["pos_tag"] == "NOUN"
    assert parsed_dict["confidence"] == 0.95

def test_sentence_analysis_metrics():
    """Test SentenceAnalysis metrics calculation"""
    tokens = [
        POSTagResult("word1", "NOUN", "NN", "Noun", 0.9, "test"),
        POSTagResult("word2", "VERB", "VB", "Verb", 0.8, "test"),
        POSTagResult("word3", "ADJ", "JJ", "Adjective", 0.95, "test")
    ]
    
    analysis = SentenceAnalysis(
        text="word1 word2 word3",
        tokens=tokens,
        model_name="test",
        processing_time=0.1,
        confidence_score=0.883  # Average of 0.9, 0.8, 0.95
    )
    
    assert analysis.confidence_score == pytest.approx(0.883, rel=1e-2)
    assert len(analysis.tokens) == 3
    assert analysis.processing_time == 0.1

# Performance tests
class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        tagger = AdvancedPOSTagger()
        tagger.models = {}  # Clear models for testing
        
        # Create large batch of texts
        texts = [f"This is test text number {i}." for i in range(100)]
        
        # Mock the tag_text method to avoid actual processing
        with patch.object(tagger, 'tag_text') as mock_tag_text:
            mock_analysis = SentenceAnalysis(
                text="test",
                tokens=[],
                model_name="test",
                processing_time=0.001,
                confidence_score=0.9
            )
            mock_tag_text.return_value = mock_analysis
            
            start_time = time.time()
            results = tagger.batch_process(texts)
            end_time = time.time()
            
            assert len(results) == 100
            assert (end_time - start_time) < 1.0  # Should be fast with mocked processing

# Test configuration
if __name__ == "__main__":
    # Run unittest tests
    unittest.main(verbosity=2)
    
    # Run pytest tests
    pytest.main([__file__, "-v"])
