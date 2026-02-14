"""
Base module for NLP learning project.
This module can contain common utilities and base classes for NLP tasks.
"""

class NLPBase:
    """
    Base class for NLP operations.
    """
    def __init__(self):
        pass
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        return text.lower().strip()
