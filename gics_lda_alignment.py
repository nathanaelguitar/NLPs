"""
Module 3.2: LDA Topic Modeling and GICS Industry Alignment

This script demonstrates how to evaluate alignment between LDA topic groupings
and GICS (Global Industry Classification Standard) industry classifications.

The workflow:
1. Infer topic probabilities for each company
2. Assign companies to their dominant topic
3. Cross-tabulate LDA topics vs GICS groups (bidirectional)
4. Analyze alignment patterns and market cap distribution
"""

import pandas as pd
import numpy as np
import tomotopy as tp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups

# Set display options
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 50)
sns.set_style('whitegrid')

print("="*80)
print("MODULE 3.2: LDA AND GICS INDUSTRY ALIGNMENT ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Simulate Company Data with Industry Labels
# ============================================================================
print("\n[1/8] Loading dataset and simulating company structure...")

# For demonstration, we'll use 20 Newsgroups as a proxy for company descriptions
# Each newsgroup category represents a different "industry"
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create a DataFrame with "companies"
df = pd.DataFrame({
    'companyid': range(len(newsgroups.data)),
    'description': newsgroups.data,
    'gics_category': newsgroups.target,
    'gics_name': [newsgroups.target_names[i] for i in newsgroups.target]
})

# Simulate market cap (larger values for certain industries)
np.random.seed(42)
df['mktcap'] = np.random.lognormal(mean=10, sigma=1.5, size=len(df))

# Simple text cleaning
def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleanDesc'] = df['description'].apply(clean_text)
df['nWords'] = df['cleanDesc'].str.split().apply(len)
df = df[df['nWords'] > 10].reset_index(drop=True)

print(f"Loaded {len(df)} 'companies' across {df['gics_name'].nunique()} 'industries'")

# ============================================================================
# STEP 2: Train LDA Model
# ============================================================================
print("\n[2/8] Training LDA model...")

# Train a simple LDA model
num_topics = 20  # Using 20 topics to match 20 "industries"
model = tp.LDAModel(k=num_topics, alpha=0.1, eta=0.01)

# Add documents
for doc in df['cleanDesc']:
    words = doc.split()
    if len(words) > 5:
        model.add_doc(words)

# Train
model.train(iter=500)
print(f"Model trained: {num_topics} topics, {len(model.docs)} documents")

# ============================================================================
# STEP 3: Infer Topic Probabilities (Embeddings)
# ============================================================================
print("\n[3/8] Inferring topic probabilities for each company...")

embeddings_list = []
for i, doc in enumerate(df['cleanDesc']):
    words = doc.split()
    if len(words) > 5:
        doc_obj = model.make_doc(words)
        topic_dist = model.infer(doc_obj)[0]
        embeddings_list.append(topic_dist)
    else:
        embeddings_list.append(np.zeros(num_topics))

# Create embeddings DataFrame
embeddings = pd.DataFrame(embeddings_list)
embeddings.columns = [f"topic{i}" for i in range(num_topics)]
embeddings['companyid'] = df['companyid'].values[:len(embeddings)]

# Merge with company data
embeddings = embeddings.merge(
    df[['companyid', 'gics_category', 'gics_name', 'mktcap']], 
    on='companyid'
)

print(f"Created embeddings matrix: {len(embeddings)} companies × {num_topics} topics")

# ============================================================================
# STEP 4: Assign Each Company to Dominant Topic
# ============================================================================
print("\n[4/8] Assigning companies to dominant topics...")

topic_cols = [f"topic{i}" for i in range(num_topics)]
embeddings['dominant_topic'] = embeddings[topic_cols].idxmax(axis=1)
embeddings['dominant_topic_prob'] = embeddings[topic_cols].max(axis=1)

print(f"Average dominant topic probability: {embeddings['dominant_topic_prob'].mean():.3f}")

# ============================================================================
# STEP 5: Name Topics Based on Top Words
# ============================================================================
print("\n[5/8] Generating topic names from top words...")

topic_names = []
for k in range(num_topics):
    top_words = [word for word, prob in model.get_topic_words(k, top_n=5)]
    topic_name = "_".join(top_words)
    topic_names.append({
        'topic': f"topic{k}",
        'topic_id': k,
        'topic_name': topic_name,
        'top_words': top_words
    })

topic_names_df = pd.DataFrame(topic_names)
embeddings = embeddings.merge(topic_names_df[['topic', 'topic_name']], 
                               left_on='dominant_topic', right_on='topic')

print("\nTop 5 topics by top words:")
for i in range(5):
    print(f"  Topic {i}: {topic_names[i]['topic_name']}")

# ============================================================================
# STEP 6: Cross-Tabulation - GICS by LDA Topics
# ============================================================================
print("\n[6/8] Cross-tabulating GICS groups by LDA topics...")

# For each GICS category, show topic distribution
gics_topic_alignment = []

for gics_name in embeddings['gics_name'].unique():
    gics_subset = embeddings[embeddings['gics_name'] == gics_name]
    topic_counts = gics_subset['topic_name'].value_counts()
    total = len(gics_subset)
    
    if total > 0:
        top_topic = topic_counts.index[0] if len(topic_counts) > 0 else "None"
        top_topic_pct = (topic_counts.iloc[0] / total * 100) if len(topic_counts) > 0 else 0
        
        gics_topic_alignment.append({
            'gics_name': gics_name,
            'num_companies': total,
            'num_topics': len(topic_counts),
            'top_topic': top_topic,
            'top_topic_pct': top_topic_pct
        })

gics_alignment_df = pd.DataFrame(gics_topic_alignment)
gics_alignment_df = gics_alignment_df.sort_values('top_topic_pct', ascending=False)

print("\nGICS Group Alignment (Top 10 by coherence):")
print(gics_alignment_df.head(10).to_string(index=False))

# ============================================================================
# STEP 7: Cross-Tabulation - LDA Topics by GICS
# ============================================================================
print("\n[7/8] Cross-tabulating LDA topics by GICS groups...")

topic_gics_alignment = []

for topic_name in embeddings['topic_name'].unique():
    topic_subset = embeddings[embeddings['topic_name'] == topic_name]
    gics_counts = topic_subset['gics_name'].value_counts()
    total = len(topic_subset)
    
    if total > 0:
        top_gics = gics_counts.index[0] if len(gics_counts) > 0 else "None"
        top_gics_pct = (gics_counts.iloc[0] / total * 100) if len(gics_counts) > 0 else 0
        
        topic_gics_alignment.append({
            'topic_name': topic_name[:50],  # Truncate for display
            'num_companies': total,
            'num_gics': len(gics_counts),
            'top_gics': top_gics,
            'top_gics_pct': top_gics_pct
        })

topic_alignment_df = pd.DataFrame(topic_gics_alignment)
topic_alignment_df = topic_alignment_df.sort_values('top_gics_pct', ascending=False)

print("\nTopic Purity (Top 10 topics by GICS concentration):")
print(topic_alignment_df.head(10).to_string(index=False))

# ============================================================================
# STEP 8: Visualizations
# ============================================================================
print("\n[8/8] Creating visualizations...")

# 1. Alignment Score Distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: GICS coherence distribution
ax1 = axes[0, 0]
ax1.hist(gics_alignment_df['top_topic_pct'], bins=20, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Top Topic Percentage (%)', fontsize=11)
ax1.set_ylabel('Number of GICS Groups', fontsize=11)
ax1.set_title('GICS Group Coherence\n(How focused is each industry on a single topic?)', 
              fontweight='bold', fontsize=12)
ax1.axvline(gics_alignment_df['top_topic_pct'].mean(), color='red', 
            linestyle='--', label=f'Mean: {gics_alignment_df["top_topic_pct"].mean():.1f}%')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Topic purity distribution
ax2 = axes[0, 1]
ax2.hist(topic_alignment_df['top_gics_pct'], bins=20, edgecolor='black', alpha=0.7, color='orange')
ax2.set_xlabel('Top GICS Percentage (%)', fontsize=11)
ax2.set_ylabel('Number of Topics', fontsize=11)
ax2.set_title('Topic Purity\n(How concentrated is each topic in a single industry?)', 
              fontweight='bold', fontsize=12)
ax2.axvline(topic_alignment_df['top_gics_pct'].mean(), color='red', 
            linestyle='--', label=f'Mean: {topic_alignment_df["top_gics_pct"].mean():.1f}%')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Company distribution by GICS
ax3 = axes[1, 0]
gics_counts = embeddings['gics_name'].value_counts().head(15)
ax3.barh(range(len(gics_counts)), gics_counts.values)
ax3.set_yticks(range(len(gics_counts)))
ax3.set_yticklabels([name[:30] for name in gics_counts.index], fontsize=9)
ax3.set_xlabel('Number of Companies', fontsize=11)
ax3.set_title('Top 15 GICS Groups by Company Count', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x')

# Bottom right: Company distribution by Topic
ax4 = axes[1, 1]
topic_counts = embeddings['topic_name'].value_counts().head(15)
ax4.barh(range(len(topic_counts)), topic_counts.values, color='orange')
ax4.set_yticks(range(len(topic_counts)))
ax4.set_yticklabels([name[:30] for name in topic_counts.index], fontsize=9)
ax4.set_xlabel('Number of Companies', fontsize=11)
ax4.set_title('Top 15 LDA Topics by Company Count', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('gics_lda_alignment_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved visualization: gics_lda_alignment_analysis.png")

# 2. Confusion Matrix-style Heatmap
print("\nCreating GICS-LDA alignment heatmap...")

# Create a cross-tabulation matrix
crosstab = pd.crosstab(
    embeddings['gics_name'], 
    embeddings['topic_name'], 
    normalize='index'
) * 100

# Select top categories for readability
top_gics = embeddings['gics_name'].value_counts().head(10).index
top_topics = embeddings['topic_name'].value_counts().head(10).index
crosstab_subset = crosstab.loc[top_gics, top_topics]

plt.figure(figsize=(14, 10))
sns.heatmap(crosstab_subset, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Percentage (%)'})
plt.title('GICS-LDA Alignment Matrix (Top 10×10)\nCell values show % of GICS group assigned to each topic', 
          fontweight='bold', fontsize=14, pad=20)
plt.xlabel('LDA Topics', fontsize=12)
plt.ylabel('GICS Groups', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('gics_lda_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved visualization: gics_lda_heatmap.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n📊 OVERALL METRICS:")
print(f"  Total companies analyzed: {len(embeddings):,}")
print(f"  Number of GICS groups: {embeddings['gics_name'].nunique()}")
print(f"  Number of LDA topics: {num_topics}")
print(f"  Average dominant topic probability: {embeddings['dominant_topic_prob'].mean():.3f}")

print(f"\n🎯 ALIGNMENT QUALITY:")
high_coherence = len(gics_alignment_df[gics_alignment_df['top_topic_pct'] >= 70])
moderate_coherence = len(gics_alignment_df[(gics_alignment_df['top_topic_pct'] >= 50) & 
                                            (gics_alignment_df['top_topic_pct'] < 70)])
low_coherence = len(gics_alignment_df[gics_alignment_df['top_topic_pct'] < 50])

print(f"  High coherence GICS groups (≥70%): {high_coherence}")
print(f"  Moderate coherence (50-70%): {moderate_coherence}")
print(f"  Low coherence (<50%): {low_coherence}")

high_purity = len(topic_alignment_df[topic_alignment_df['top_gics_pct'] >= 70])
moderate_purity = len(topic_alignment_df[(topic_alignment_df['top_gics_pct'] >= 50) & 
                                          (topic_alignment_df['top_gics_pct'] < 70)])
low_purity = len(topic_alignment_df[topic_alignment_df['top_gics_pct'] < 50])

print(f"\n  High purity topics (≥70%): {high_purity}")
print(f"  Moderate purity (50-70%): {moderate_purity}")
print(f"  Low purity (<50%): {low_purity}")

print("\n" + "="*80)
print("💡 KEY TAKEAWAYS:")
print("="*80)
print("""
1. High coherence GICS groups have distinctive vocabulary that maps cleanly to topics
2. Low coherence groups indicate either diverse business models or generic language
3. High purity topics correspond to specific industries (pure topics)
4. Low purity topics capture cross-industry themes (mixed topics)
5. The gap between coherence and purity reveals where LDA cuts across GICS boundaries
""")

print("\n✓ Analysis complete!")
print("Files saved:")
print("  - gics_lda_alignment_analysis.png")
print("  - gics_lda_heatmap.png")
