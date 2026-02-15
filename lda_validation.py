"""
LDA Validation Utilities

Comprehensive validation functions for LDA topic models using tomotopy.
Based on best practices for quantitative topic model evaluation.

Usage:
    from lda_validation import *
    
    # After training your model:
    validate_model(model, output_prefix="my_analysis")
"""

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def find_term_topic(model, term, top_n=500):
    """
    Find the topic where a term has the highest probability P(word | topic).
    
    Args:
        model: Trained tomotopy LDA model
        term: Word to search for
        top_n: Number of top words to check per topic
    
    Returns:
        (best_topic_id, best_probability) or (None, 0.0) if not found
    
    Example:
        >>> topic_id, prob = find_term_topic(model, "computer")
        >>> print(f"'computer' peaks at topic {topic_id} with P={prob:.5f}")
    """
    best_topic = None
    best_prob = 0.0
    
    for k in range(model.k):
        words = dict(model.get_topic_words(k, top_n=top_n))
        if term in words and words[term] > best_prob:
            best_prob = words[term]
            best_topic = k
    
    return best_topic, best_prob


def term_weights_by_topic(model, terms, topn=500):
    """
    Get P(word | topic) for multiple terms across all topics.
    
    Args:
        model: Trained tomotopy LDA model
        terms: List of terms to analyze
        topn: Number of top words to check per topic
    
    Returns:
        Dict mapping topic_id -> {term: probability}
    
    Example:
        >>> terms = ["computer", "software", "hardware"]
        >>> weights = term_weights_by_topic(model, terms)
        >>> print(weights[0])  # Show term weights for topic 0
    """
    out = {}
    for k in range(model.k):
        w = dict(model.get_topic_words(k, top_n=topn))
        hits = {t: w.get(t, 0.0) for t in terms if t in w}
        if hits:
            out[k] = hits
    return out


def calculate_topic_separation(model):
    """
    Calculate average cosine similarity between topics.
    
    Lower values indicate better topic separation.
    
    Args:
        model: Trained tomotopy LDA model
    
    Returns:
        (avg_similarity, similarity_matrix)
    
    Interpretation:
        < 0.3: Topics are highly distinct (excellent)
        0.3-0.5: Moderate separation (good)
        > 0.5: Topics are overlapping (needs improvement)
    
    Example:
        >>> avg_sim, sim_matrix = calculate_topic_separation(model)
        >>> print(f"Average similarity: {avg_sim:.4f}")
    """
    # Get topic-word distributions
    topic_dists = np.stack([
        model.get_topic_word_dist(k)
        for k in range(model.k)
    ])
    
    # Calculate cosine similarity
    sim_matrix = cosine_similarity(topic_dists)
    
    # Calculate average off-diagonal similarity
    k = model.k
    avg_sim = (sim_matrix.sum() - k) / (k * (k - 1))
    
    return avg_sim, sim_matrix


def plot_topic_similarity_matrix(model, save_path=None):
    """
    Visualize topic similarity as a heatmap.
    
    Args:
        model: Trained tomotopy LDA model
        save_path: Optional path to save the figure
    
    Example:
        >>> plot_topic_similarity_matrix(model, "topic_similarity.png")
    """
    avg_sim, sim_matrix = calculate_topic_separation(model)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Topic Similarity Matrix\n(Avg Similarity: {avg_sim:.4f})', 
              fontweight='bold', fontsize=14)
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Topic ID', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved similarity matrix to {save_path}")
    
    plt.show()


def analyze_term_distribution(model, terms, top_topics=3):
    """
    Print detailed term distribution across topics.
    
    Args:
        model: Trained tomotopy LDA model
        terms: List of terms to analyze
        top_topics: Number of top topics to show per term
    
    Example:
        >>> terms = ["computer", "game", "medical"]
        >>> analyze_term_distribution(model, terms)
    """
    print("="*70)
    print("TERM DISTRIBUTION ACROSS TOPICS")
    print("="*70)
    
    tw = term_weights_by_topic(model, terms)
    
    for term in terms:
        # Rank topics by this term's probability
        ranked = sorted(
            [(k, tw[k].get(term, 0.0)) for k in tw if term in tw[k]],
            key=lambda x: x[1],
            reverse=True
        )
        
        if ranked:
            print(f"\n{term}:")
            for k, val in ranked[:top_topics]:
                print(f"  Topic {k:2d}: {val:.5f}")
                top_words = ', '.join([w for w, p in model.get_topic_words(k, top_n=5)])
                print(f"            [{top_words}]")
        else:
            print(f"\n{term}: Not found in top words")


def get_most_similar_topic_pairs(model, n_pairs=5):
    """
    Find the most similar topic pairs.
    
    Args:
        model: Trained tomotopy LDA model
        n_pairs: Number of pairs to return
    
    Returns:
        List of (topic_i, topic_j, similarity) tuples
    
    Example:
        >>> pairs = get_most_similar_topic_pairs(model)
        >>> for i, j, sim in pairs:
        ...     print(f"Topics {i} & {j}: similarity={sim:.4f}")
    """
    _, sim_matrix = calculate_topic_separation(model)
    k = model.k
    
    similar_pairs = []
    for i in range(k):
        for j in range(i+1, k):
            similar_pairs.append((i, j, sim_matrix[i, j]))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs[:n_pairs]


def print_validation_report(model, coherence_score=None, perplexity=None):
    """
    Print comprehensive validation report.
    
    Args:
        model: Trained tomotopy LDA model
        coherence_score: Optional coherence score
        perplexity: Optional perplexity (defaults to model.perplexity)
    
    Example:
        >>> from your_coherence_function import getCoherence
        >>> coherence = getCoherence(model, ncore=4)
        >>> print_validation_report(model, coherence_score=coherence)
    """
    print("\n" + "="*70)
    print("LDA MODEL VALIDATION REPORT")
    print("="*70)
    
    # Basic stats
    print("\n📊 MODEL CONFIGURATION")
    print(f"  Number of topics (k):        {model.k}")
    print(f"  Documents processed:         {len(model.docs):,}")
    print(f"  Vocabulary size:             {len(model.vocabs):,}")
    print(f"  Used vocabulary:             {len(model.used_vocabs):,}")
    
    # Performance metrics
    print("\n📈 PERFORMANCE METRICS")
    if perplexity is None:
        perplexity = model.perplexity
    print(f"  Perplexity (lower=better):   {perplexity:.4f}")
    
    if coherence_score is not None:
        print(f"  Coherence (higher=better):   {coherence_score:.4f}")
        if coherence_score > 0.4:
            print("    ✓ EXCELLENT coherence")
        elif coherence_score > 0.3:
            print("    ✓ GOOD coherence")
        else:
            print("    ⚠ Coherence needs improvement")
    
    # Topic separation
    avg_sim, _ = calculate_topic_separation(model)
    print(f"  Topic separation (avg sim):  {avg_sim:.4f}")
    if avg_sim < 0.3:
        print("    ✓ EXCELLENT separation")
    elif avg_sim < 0.5:
        print("    ✓ GOOD separation")
    else:
        print("    ⚠ Topics may be overlapping")
    
    # Most similar pairs
    print("\n🔍 MOST SIMILAR TOPIC PAIRS")
    pairs = get_most_similar_topic_pairs(model, n_pairs=3)
    for i, j, sim in pairs:
        print(f"\n  Topics {i} & {j} (similarity: {sim:.4f})")
        words_i = ', '.join([w for w, p in model.get_topic_words(i, top_n=5)])
        words_j = ', '.join([w for w, p in model.get_topic_words(j, top_n=5)])
        print(f"    Topic {i}: {words_i}")
        print(f"    Topic {j}: {words_j}")
    
    print("\n" + "="*70)


def validate_model(model, coherence_score=None, terms_to_check=None, 
                   output_prefix=None):
    """
    Run full validation suite and optionally save results.
    
    Args:
        model: Trained tomotopy LDA model
        coherence_score: Optional coherence score
        terms_to_check: Optional list of terms to analyze
        output_prefix: Optional prefix for saved files
    
    Example:
        >>> validate_model(
        ...     model,
        ...     coherence_score=0.45,
        ...     terms_to_check=["computer", "medical", "space"],
        ...     output_prefix="newsgroups"
        ... )
    """
    # Print validation report
    print_validation_report(model, coherence_score=coherence_score)
    
    # Analyze terms if provided
    if terms_to_check:
        print("\n")
        analyze_term_distribution(model, terms_to_check)
    
    # Plot similarity matrix
    if output_prefix:
        plot_path = f"{output_prefix}_topic_similarity.png"
        plot_topic_similarity_matrix(model, save_path=plot_path)
    else:
        plot_topic_similarity_matrix(model)
    
    # Save validation results
    if output_prefix:
        results = {
            'num_topics': model.k,
            'num_docs': len(model.docs),
            'vocab_size': len(model.vocabs),
            'perplexity': model.perplexity,
            'coherence': coherence_score,
        }
        
        avg_sim, sim_matrix = calculate_topic_separation(model)
        results['avg_topic_similarity'] = avg_sim
        results['similarity_matrix'] = sim_matrix
        
        results_path = f"{output_prefix}_validation_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Saved validation results to {results_path}")


if __name__ == "__main__":
    print("LDA Validation Utilities")
    print("Import this module to use validation functions:")
    print("  from lda_validation import *")
