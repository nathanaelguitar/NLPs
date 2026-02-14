"""
NLP Learning Script
This script demonstrates the usage of various NLP libraries for topic modeling and analysis.
"""

import importlib
import Base
import tomotopy as tp
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import time
from gensim.models import CoherenceModel
import collections
import gensim
import pickle


def main():
    """
    Main function to demonstrate NLP library usage.
    """
    print("NLP Learning Environment Setup Complete!")
    print("\nAvailable libraries:")
    print("- Base (custom module)")
    print("- tomotopy (topic modeling)")
    print("- matplotlib (visualization)")
    print("- wordcloud (word cloud generation)")
    print("- gensim (topic modeling and NLP)")
    print("- pickle (data serialization)")
    print("- collections (data structures)")
    print("- time (timing operations)")
    
    # Demonstrate Base module
    base = Base.NLPBase()
    sample_text = "  HELLO WORLD  "
    processed = base.preprocess_text(sample_text)
    print(f"\nBase module test: '{sample_text}' -> '{processed}'")
    
    # Demonstrate simple topic modeling with tomotopy
    print("\nTomotopy version:", tp.isa)
    print("Gensim version:", gensim.__version__)
    
    # Example: Create a simple LDA model
    mdl = tp.LDAModel(k=5)  # 5 topics
    print(f"Created LDA model with {mdl.k} topics")
    
    print("\nSetup complete! Ready for NLP learning.")


if __name__ == "__main__":
    main()
