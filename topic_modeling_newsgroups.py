"""
Topic Modeling on 20 Newsgroups Dataset
Using Tomotopy LDA with hyperparameter tuning and pyLDAvis visualization
"""

import pandas as pd
import numpy as np
import tomotopy as tp
import gensim
import gensim.corpora
import gensim.models.coherencemodel
import collections
import pickle
import time
import matplotlib.pyplot as plt
import pyLDAvis
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

print("=" * 80)
print("TOPIC MODELING PROJECT - 20 NEWSGROUPS DATASET")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================
print("\n[1/7] Loading 20 Newsgroups dataset...")

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data
categories = newsgroups.target_names

print(f"Loaded {len(docs)} documents")
print(f"Categories: {len(categories)}")

# Create DataFrame
df = pd.DataFrame({
    'text': docs,
    'category': [categories[i] for i in newsgroups.target]
})

print("\n[2/7] Preprocessing text...")

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

df['cleanText'] = df['text'].apply(clean_text)

# Filter documents
df['nWords'] = df['cleanText'].str.split().apply(len)
df = df[df['nWords'] > 10]  # Keep documents with more than 10 words
df = df.reset_index(drop=True)

print(f"After filtering: {len(df)} documents")
print(f"Average words per document: {df['nWords'].mean():.1f}")

# ============================================================================
# STEP 2: HELPER FUNCTIONS
# ============================================================================

def getCoherence(model, ncore=4):
    """Calculate coherence score for a tomotopy model"""
    topics = []
    for k in range(model.k):
        word_probs = model.get_topic_words(k)
        topics.append([word for word, prob in word_probs])
        
    texts = []
    corpus = []
    for doc in model.docs:
        words = [model.vocabs[token_id] for token_id in doc.words]
        texts.append(words)
        freqs = list(collections.Counter(doc.words).items())
        corpus.append(freqs)
    
    id2word = dict(enumerate(model.vocabs))
    dictionary = gensim.corpora.dictionary.Dictionary.from_corpus(
        corpus, id2word
    )
    
    cm = gensim.models.coherencemodel.CoherenceModel(
        topics=topics,
        texts=texts,
        corpus=corpus,
        dictionary=dictionary,
        processes=ncore
    )
    
    return cm.get_coherence()

def buildModel(df, hP, txt_col, k, alpha, eta):
    """Build and train an LDA model"""
    model = tp.LDAModel(
        alpha=alpha,      
        eta=eta,   
        k=k
    )

    df_split = [doc.split() for doc in df[txt_col].astype(str)]
    for doc in df_split: 
        model.add_doc(doc)

    model.train(iter=hP['train_iter'])

    return model

# ============================================================================
# STEP 3: HYPERPARAMETER TUNING - NUMBER OF TOPICS (K)
# ============================================================================
print("\n[3/7] Tuning number of topics (k)...")

hP = {
    'train_iter': 500,
    'n_cores': 4,
}

coherence_k = []
perplexity_k = []
ks = [10, 15, 20, 25, 30]
alphas = [0.1]
etas = [0.01]

for k in ks:
    for alpha in alphas:
        for eta in etas:
            print(f"  Testing k={k}, alpha={alpha}, eta={eta}...")
            ldaModel = buildModel(df, hP, 'cleanText', k, alpha, eta)
            p = ldaModel.perplexity
            c = getCoherence(ldaModel, 4)
            coherence_k.append([k, alpha, eta, c])
            perplexity_k.append([k, alpha, eta, p])
            print(f"    Perplexity: {p:.4f}, Coherence: {c:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

X = [x[0] for x in coherence_k]
Y = [y[3] for y in coherence_k]
ax1.plot(X, Y, marker='o', linewidth=2, markersize=8)
ax1.set_title('Topic Coherence by Number of Topics', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Topics (k)', fontsize=11)
ax1.set_ylabel('Coherence Score', fontsize=11)
ax1.grid(True, alpha=0.3)

X = [x[0] for x in perplexity_k]
Y = [y[3] for y in perplexity_k]
ax2.plot(X, Y, marker='o', linewidth=2, markersize=8, color='orange')
ax2.set_title('Model Perplexity by Number of Topics', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Topics (k)', fontsize=11)
ax2.set_ylabel('Perplexity', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_k.png', dpi=150, bbox_inches='tight')
plt.show()

# Find best k
best_k_idx = np.argmax([x[3] for x in coherence_k])
best_k = coherence_k[best_k_idx][0]
print(f"\nBest k (by coherence): {best_k}")

# ============================================================================
# STEP 4: HYPERPARAMETER TUNING - ALPHA
# ============================================================================
print("\n[4/7] Tuning alpha (document-topic distribution)...")

coherence_alphas = []
perplexity_alphas = []
ks = [best_k]
alphas = [0.01, 0.1, 0.5, 1.0, 2.0]
etas = [0.01]

for k in ks:
    for alpha in alphas:
        for eta in etas:
            print(f"  Testing k={k}, alpha={alpha}, eta={eta}...")
            ldaModel = buildModel(df, hP, 'cleanText', k, alpha, eta)
            p = ldaModel.perplexity
            c = getCoherence(ldaModel, 4)
            coherence_alphas.append([k, alpha, eta, c])
            perplexity_alphas.append([k, alpha, eta, p])
            print(f"    Perplexity: {p:.4f}, Coherence: {c:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

X = [x[1] for x in coherence_alphas]
Y = [y[3] for y in coherence_alphas]
ax1.plot(X, Y, marker='o', linewidth=2, markersize=8)
ax1.set_title('Topic Coherence by Alpha', fontsize=12, fontweight='bold')
ax1.set_xlabel('Alpha', fontsize=11)
ax1.set_ylabel('Coherence Score', fontsize=11)
ax1.grid(True, alpha=0.3)

X = [x[1] for x in perplexity_alphas]
Y = [y[3] for y in perplexity_alphas]
ax2.plot(X, Y, marker='o', linewidth=2, markersize=8, color='orange')
ax2.set_title('Model Perplexity by Alpha', fontsize=12, fontweight='bold')
ax2.set_xlabel('Alpha', fontsize=11)
ax2.set_ylabel('Perplexity', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_alpha.png', dpi=150, bbox_inches='tight')
plt.show()

# Find best alpha
best_alpha_idx = np.argmax([x[3] for x in coherence_alphas])
best_alpha = coherence_alphas[best_alpha_idx][1]
print(f"\nBest alpha (by coherence): {best_alpha}")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING - ETA
# ============================================================================
print("\n[5/7] Tuning eta (topic-word distribution)...")

coherence_etas = []
perplexity_etas = []
ks = [best_k]
alphas = [best_alpha]
etas = [0.001, 0.01, 0.05, 0.1, 0.2]

for k in ks:
    for alpha in alphas:
        for eta in etas:
            print(f"  Testing k={k}, alpha={alpha}, eta={eta}...")
            ldaModel = buildModel(df, hP, 'cleanText', k, alpha, eta)
            p = ldaModel.perplexity
            c = getCoherence(ldaModel, 4)
            coherence_etas.append([k, alpha, eta, c])
            perplexity_etas.append([k, alpha, eta, p])
            print(f"    Perplexity: {p:.4f}, Coherence: {c:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

X = [x[2] for x in coherence_etas]
Y = [y[3] for y in coherence_etas]
ax1.plot(X, Y, marker='o', linewidth=2, markersize=8)
ax1.set_title('Topic Coherence by Eta', fontsize=12, fontweight='bold')
ax1.set_xlabel('Eta', fontsize=11)
ax1.set_ylabel('Coherence Score', fontsize=11)
ax1.grid(True, alpha=0.3)

X = [x[2] for x in perplexity_etas]
Y = [y[3] for y in perplexity_etas]
ax2.plot(X, Y, marker='o', linewidth=2, markersize=8, color='orange')
ax2.set_title('Model Perplexity by Eta', fontsize=12, fontweight='bold')
ax2.set_xlabel('Eta', fontsize=11)
ax2.set_ylabel('Perplexity', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_eta.png', dpi=150, bbox_inches='tight')
plt.show()

# Find best eta
best_eta_idx = np.argmax([x[3] for x in coherence_etas])
best_eta = coherence_etas[best_eta_idx][2]
print(f"\nBest eta (by coherence): {best_eta}")

# ============================================================================
# STEP 6: BUILD FINAL MODEL WITH OPTIMAL PARAMETERS
# ============================================================================
print("\n[6/7] Building final model with optimal parameters...")
print(f"  k={best_k}, alpha={best_alpha}, eta={best_eta}")

hP_final = {
    'train_iter': 1000,
    'n_cores': 4,
    'eta': best_eta,
    'alpha': best_alpha,
    'k': best_k
}

def buildFinalModel(df, hP, txt_col):
    """Build final model with parameters from hP dict"""
    model = tp.LDAModel(
        alpha=hP['alpha'],      
        eta=hP['eta'],   
        k=hP['k']
    )

    df_split = [doc.split() for doc in df[txt_col].astype(str)]
    for doc in df_split: 
        model.add_doc(doc)

    model.train(iter=hP['train_iter'])

    return model

final_model = buildFinalModel(df, hP_final, 'cleanText')

print("\n### MODEL EVALUATION ###")
print(f"Perplexity: {final_model.perplexity:.4f}")
print(f"Coherence: {getCoherence(final_model, 4):.4f}")

print("\n### TOP WORDS PER TOPIC ###")
for k in range(final_model.k):
    print(f'\nTopic #{k}:')
    words = final_model.get_topic_words(k, top_n=10)
    print(', '.join([word for word, prob in words]))

# Save the model
print("\nSaving model...")
final_model.save("newsgroups_lda_final.mdl")
pickle.dump(df, open("newsgroups_df.p", "wb"))
print("Model saved as 'newsgroups_lda_final.mdl'")
print("DataFrame saved as 'newsgroups_df.p'")

# ============================================================================
# STEP 7: PYLDAVIS VISUALIZATION
# ============================================================================
print("\n[7/7] Creating pyLDAvis visualization...")

# Prepare data for pyLDAvis
topic_term_dists = np.stack([final_model.get_topic_word_dist(k) for k in range(final_model.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in final_model.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in final_model.docs])
vocab = list(final_model.used_vocabs)
term_frequency = final_model.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists, 
    doc_topic_dists, 
    doc_lengths, 
    vocab, 
    term_frequency,
    start_index=0,
    sort_topics=False
)

pyLDAvis.save_html(prepared_data, 'newsgroups_ldavis.html')
print("Interactive visualization saved as 'newsgroups_ldavis.html'")
print("Open this file in a web browser to explore the topics!")

print("\n" + "=" * 80)
print("TOPIC MODELING COMPLETE!")
print("=" * 80)
print("\nFiles created:")
print("  - hyperparameter_tuning_k.png")
print("  - hyperparameter_tuning_alpha.png")
print("  - hyperparameter_tuning_eta.png")
print("  - newsgroups_lda_final.mdl")
print("  - newsgroups_df.p")
print("  - newsgroups_ldavis.html")
print("\nOptimal parameters:")
print(f"  k (topics): {best_k}")
print(f"  alpha: {best_alpha}")
print(f"  eta: {best_eta}")
