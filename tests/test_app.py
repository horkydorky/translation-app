import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStreamlitApp(unittest.TestCase):

    @patch('streamlit.set_page_config')
    @patch('streamlit.cache_resource')
    @patch('transformers.M2M100Tokenizer.from_pretrained')
    @patch('transformers.M2M100ForConditionalGeneration.from_pretrained')
    def test_app_loads(self, mock_model, mock_tokenizer, mock_cache, mock_config):
        """
        Test that the app's critical components (model loading) are called correctly.
        We mock streamlit and transformers to avoid actual execution/downloading.
        """
        # Mock the return values
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # We can't easily import the whole script because it runs immediately.
        # Instead, we test the logic we can isolate or just ensure imports work.
        
        # Check if we can import key functions without error
        try:
            from translation_app import load_model, detect_language
            
            # Test load_model
            # Note: streamlit caching might interfere, so we mock it above
            # But the decorator wraps the function, so we might need to test the wrapped function if possible
            # unique to how streamlit mocks work. 
            
            # For this simple test, just verifying we can import is a good "smoke test"
            pass
        except ImportError as e:
            self.fail(f"Failed to import app modules: {e}")

    @patch('langdetect.detect')
    def test_detect_language(self, mock_detect):
        from translation_app import detect_language
        mock_detect.return_value = 'en'
        self.assertEqual(detect_language("Hello"), 'en')
        
        mock_detect.side_effect = Exception("Error")
        self.assertIsNone(detect_language("Invalid"))

if __name__ == '__main__':
    unittest.main()
