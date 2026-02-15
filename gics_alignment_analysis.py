"""
Module 3.2: LDA Topic Modeling and GICS Industry Alignment

This script demonstrates how to evaluate alignment between LDA topic groupings
and GICS (Global Industry Classification Standard) industry groups.

Prerequisites:
- Trained LDA model (e.g., ldaIndGrpCleanFinal.mdl)
- Company data with GICS codes and cleaned descriptions (e.g., longD_unit2.p)
- Base.py helper module (provided by course)

Dataset: Middle 1000 companies by market capitalization
Model: 24-topic LDA model
"""

import pandas as pd
import numpy as np
import tomotopy as tp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')

print("="*80)
print("MODULE 3.2: LDA AND GICS INDUSTRY ALIGNMENT ANALYSIS")
print("="*80)


# ============================================================================
# GICS INDUSTRY GROUP LOOKUP TABLE
# ============================================================================

def get_gics_industry_names():
    """
    Returns DataFrame mapping GICS industry group codes to readable names.
    These 24 groups cover the Middle 1000 dataset.
    """
    indGrpNames = pd.DataFrame([
        [1010, 'Energy'],
        [1510, 'Materials'],
        [2010, 'Capital Goods'],
        [2020, 'Commercial & Professional Services'],
        [2030, 'Transportation'],
        [2510, 'Automobiles & Components'],
        [2520, 'Consumer Durables & Apparel'],
        [2530, 'Consumer Services'],
        [2550, 'Retailing'],
        [3010, 'Food & Staples Retailing'],
        [3020, 'Food, Beverage & Tobacco'],
        [3030, 'Household & Personal Products'],
        [3510, 'Health Care Equipment & Services'],
        [3520, 'Pharmaceuticals, Biotechnology & Life Sciences'],
        [4010, 'Banks'],
        [4020, 'Diversified Financials'],
        [4030, 'Insurance'],
        [4510, 'Software & Services'],
        [4520, 'Technology Hardware & Equipment'],
        [4530, 'Semiconductors & Semiconductor Equipment'],
        [5010, 'Telecommunication Services'],
        [5020, 'Media & Entertainment'],
        [5510, 'Utilities'],
        [6010, 'Real Estate']
    ])
    indGrpNames.columns = ['gicsIndGrp', 'indGrp']
    return indGrpNames


# ============================================================================
# STEP 1: LOAD DATA AND MODEL
# ============================================================================

def load_data_and_model(data_path, model_path):
    """
    Load company descriptions and trained LDA model.
    
    Args:
        data_path: Path to pickled company data (e.g., "longD_unit2.p")
        model_path: Path to trained LDA model (e.g., "ldaIndGrpCleanFinal.mdl")
    
    Returns:
        (longD DataFrame, ldaModel)
    """
    print("\n[1/8] Loading data and model...")
    
    # Load company data
    longD = pickle.load(open(data_path, "rb"))
    longD = longD.sort_values(by='mktcap', ascending=False)
    longD = longD.reset_index(drop=True)
    longD = longD[longD['gics'] > 0]  # Filter to companies with GICS codes
    longD['nWords'] = longD['cleanDesc'].str.split().apply(len)
    longD = longD[longD['nWords'] > 5]  # Keep descriptions with >5 words
    longD['gicsIndGrp'] = (longD['gicsInd'] / 100).astype(int)
    
    # Load LDA model
    ldaModel = tp.LDAModel.load(model_path)
    
    print(f"✓ Loaded {len(longD)} companies")
    print(f"✓ Model has {ldaModel.k} topics")
    
    return longD, ldaModel


# ============================================================================
# STEP 2: INFER TOPIC PROBABILITIES (EMBEDDINGS)
# ============================================================================

def infer_topic_embeddings(longD, ldaModel):
    """
    Infer topic probability distributions for each company.
    
    Args:
        longD: Company DataFrame with cleanDesc column
        ldaModel: Trained LDA model
    
    Returns:
        DataFrame with topic probabilities for each company
    """
    print("\n[2/8] Inferring topic probabilities...")
    
    companyId = longD['companyid'].tolist()
    cleanDesc = [doc.split() for doc in longD['cleanDesc'].astype(str)]
    
    # Infer topic distributions
    embeddings = [ldaModel.infer(ldaModel.make_doc(desc)) for desc in cleanDesc]
    
    # Convert to DataFrame
    embeddings = pd.DataFrame([x[0] for x in embeddings])
    embeddings.columns = [f"topic{x}" for x in range(ldaModel.k)]
    embeddings.index = companyId
    embeddings = embeddings.reset_index()
    embeddings = embeddings.rename(columns={'index': 'companyid'})
    
    # Merge with company metadata
    embeddings = embeddings.merge(
        longD[['companyid', 'gvkey', 'compustat_name', 'ticker', 'mktcap',
               'gics', 'gicsIndGrp', 'gicsInd', 'gicsSubind']],
        on='companyid'
    )
    
    print(f"✓ Generated embeddings for {len(embeddings)} companies")
    print(f"✓ Each company has {ldaModel.k} topic probabilities")
    
    return embeddings


# ============================================================================
# STEP 3: ASSIGN DOMINANT TOPIC AND ADD METADATA
# ============================================================================

def assign_dominant_topics(embeddings, ldaModel):
    """
    Assign each company to its highest-probability topic and add topic names.
    
    Args:
        embeddings: DataFrame with topic probabilities
        ldaModel: Trained LDA model
    
    Returns:
        Enhanced embeddings DataFrame with topic assignments and names
    """
    print("\n[3/8] Assigning dominant topics...")
    
    # Find dominant topic for each company
    topic_cols = [f"topic{x}" for x in range(ldaModel.k)]
    maxtopic = embeddings[topic_cols].idxmax(axis=1)
    embeddings = embeddings.merge(maxtopic.rename('topic'), left_index=True, right_index=True)
    
    # Create topic names from top words
    topicnames = []
    for k in range(ldaModel.k):
        nm = ""
        for word, prob in ldaModel.get_topic_words(k, top_n=5):
            nm += word + "_"
        topicnames.append([
            f"topic{k}",
            k,
            nm,
            [w for w, p in ldaModel.get_topic_words(k, top_n=10)]
        ])
    
    topicnames = pd.DataFrame(topicnames)
    topicnames.columns = ['topic', 'topicn', 'topic_name', 'topic_words']
    embeddings = embeddings.merge(topicnames, on='topic')
    
    # Add GICS industry group names
    indGrpNames = get_gics_industry_names()
    embeddings = embeddings.merge(indGrpNames, on='gicsIndGrp')
    
    print(f"✓ Assigned {len(embeddings)} companies to dominant topics")
    
    return embeddings, topicnames


# ============================================================================
# STEP 4: ANALYZE COMPANY COUNTS BY GICS AND TOPICS
# ============================================================================

def analyze_company_counts(embeddings):
    """
    Count companies by GICS group and LDA topic.
    """
    print("\n[4/8] Analyzing company distributions...")
    
    # Counts by GICS
    gics_counts = embeddings.groupby('indGrp').size().sort_values(ascending=False)
    print("\nTop 5 GICS Industry Groups by Company Count:")
    print(gics_counts.head())
    
    # Counts by LDA topic
    topic_counts = embeddings.groupby('topic_name').size().sort_values(ascending=False)
    print("\nTop 5 LDA Topics by Company Count:")
    print(topic_counts.head())
    
    return gics_counts, topic_counts


# ============================================================================
# STEP 5: CROSS-TABULATE LDA TOPICS BY GICS GROUP
# ============================================================================

def cross_tab_gics_to_lda(embeddings, indGrpNames):
    """
    For each GICS group, show which LDA topics its companies belong to.
    
    This answers: "How thematically coherent is each GICS group?"
    """
    print("\n[5/8] Cross-tabulating GICS → LDA...")
    print("\nFor each GICS group, showing topic distribution:")
    print("="*80)
    
    results = []
    
    for idx, row in indGrpNames.iterrows():
        group_data = embeddings[embeddings.gicsIndGrp == row.gicsIndGrp]
        
        if len(group_data) == 0:
            continue
            
        ntopics = group_data.groupby('topic_name').size().reset_index(name='n')
        ntopics['pct'] = round(100.0 * ntopics['n'] / ntopics['n'].sum(), 1)
        ntopics = ntopics.sort_values(by='n', ascending=False)
        ntopics['gicsIndGrp'] = row.gicsIndGrp
        ntopics['indGrp'] = row.indGrp
        
        print(f"\n{row.indGrp} ({row.gicsIndGrp}) - {len(group_data)} companies:")
        print(ntopics.head(3)[['topic_name', 'n', 'pct']].to_string(index=False))
        
        # Store top concentration for summary
        top_pct = ntopics.iloc[0]['pct'] if len(ntopics) > 0 else 0
        results.append([row.indGrp, len(group_data), top_pct])
    
    results_df = pd.DataFrame(results, columns=['GICS Group', 'Companies', 'Top Topic %'])
    results_df = results_df.sort_values('Top Topic %', ascending=False)
    
    print("\n" + "="*80)
    print("SUMMARY: GICS Group Coherence (% in dominant topic)")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


# ============================================================================
# STEP 6: CROSS-TABULATE GICS GROUPS BY LDA TOPIC
# ============================================================================

def cross_tab_lda_to_gics(embeddings, topicnames):
    """
    For each LDA topic, show which GICS groups are represented.
    
    This answers: "Does each topic correspond to single or multiple industries?"
    """
    print("\n[6/8] Cross-tabulating LDA → GICS...")
    print("\nFor each LDA topic, showing GICS distribution:")
    print("="*80)
    
    results = []
    
    for idx, row in topicnames.iterrows():
        topic_data = embeddings[embeddings.topic == row.topic]
        
        if len(topic_data) == 0:
            continue
            
        ngroups = topic_data.groupby('indGrp').size().reset_index(name='n')
        ngroups['pct'] = round(100.0 * ngroups['n'] / ngroups['n'].sum(), 1)
        ngroups = ngroups.sort_values(by='n', ascending=False)
        
        print(f"\n{row.topic_name[:50]} - {len(topic_data)} companies:")
        print(ngroups.head(3)[['indGrp', 'n', 'pct']].to_string(index=False))
        
        # Check if topic is pure (>80% from one GICS group)
        top_pct = ngroups.iloc[0]['pct'] if len(ngroups) > 0 else 0
        purity = "Pure" if top_pct > 80 else "Mixed"
        results.append([row.topic_name[:30], len(topic_data), top_pct, purity])
    
    results_df = pd.DataFrame(results, columns=['Topic', 'Companies', 'Top GICS %', 'Type'])
    results_df = results_df.sort_values('Top GICS %', ascending=False)
    
    print("\n" + "="*80)
    print("SUMMARY: Topic Purity (% from dominant GICS group)")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


# ============================================================================
# STEP 7: MARKET CAPITALIZATION ANALYSIS
# ============================================================================

def analyze_market_cap(embeddings, topicnames):
    """
    Calculate market cap distribution by GICS and by LDA topics.
    
    GICS uses hard assignment (each company in one group).
    LDA uses soft assignment (weighted by topic probabilities).
    """
    print("\n[7/8] Analyzing market capitalization distribution...")
    
    totcap = embeddings.mktcap.sum()
    
    # Market cap by GICS (hard assignment)
    mcapGics = embeddings.groupby('indGrp').mktcap.sum() / totcap * 100
    mcapGics = mcapGics.sort_values(ascending=False)
    
    print("\nTop 5 GICS Groups by Market Cap %:")
    print(mcapGics.head().round(2))
    
    # Market cap by LDA topic (soft assignment, weighted by probability)
    mcapTopic = []
    for idx, t in topicnames.iterrows():
        topic_col = t.topic
        dfMcap = embeddings[topic_col] * embeddings['mktcap']
        mcap_pct = round(dfMcap.sum() / totcap * 100, 2)
        mcapTopic.append([t.topic_name[:30], mcap_pct])
    
    mcapTopic = pd.DataFrame(mcapTopic, columns=['Topic', 'Market Cap %'])
    mcapTopic = mcapTopic.sort_values('Market Cap %', ascending=False)
    
    print("\nTop 5 LDA Topics by Market Cap %:")
    print(mcapTopic.head().to_string(index=False))
    
    return mcapGics, mcapTopic


# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================

def create_visualizations(embeddings, gics_coherence, topic_purity):
    """
    Create visualizations of alignment patterns.
    """
    print("\n[8/8] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. GICS Group Coherence
    top_gics = gics_coherence.head(10)
    axes[0, 0].barh(range(len(top_gics)), top_gics['Top Topic %'].values)
    axes[0, 0].set_yticks(range(len(top_gics)))
    axes[0, 0].set_yticklabels(top_gics['GICS Group'].values)
    axes[0, 0].set_xlabel('% in Dominant Topic')
    axes[0, 0].set_title('GICS Group Coherence\n(% of companies in top LDA topic)', 
                          fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0, 0].legend()
    
    # 2. Topic Purity
    top_topics = topic_purity.head(10)
    colors = ['green' if t == 'Pure' else 'orange' for t in top_topics['Type']]
    axes[0, 1].barh(range(len(top_topics)), top_topics['Top GICS %'].values, color=colors)
    axes[0, 1].set_yticks(range(len(top_topics)))
    axes[0, 1].set_yticklabels(top_topics['Topic'].values)
    axes[0, 1].set_xlabel('% from Dominant GICS')
    axes[0, 1].set_title('LDA Topic Purity\n(% from top GICS group)', 
                          fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0, 1].legend()
    
    # 3. Company counts by GICS
    gics_counts = embeddings.groupby('indGrp').size().sort_values(ascending=False).head(10)
    axes[1, 0].bar(range(len(gics_counts)), gics_counts.values)
    axes[1, 0].set_xticks(range(len(gics_counts)))
    axes[1, 0].set_xticklabels(gics_counts.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Number of Companies')
    axes[1, 0].set_title('Top 10 GICS Groups by Company Count', fontweight='bold')
    
    # 4. Company counts by topic
    topic_counts = embeddings.groupby('topic_name').size().sort_values(ascending=False).head(10)
    axes[1, 1].bar(range(len(topic_counts)), topic_counts.values, color='orange')
    axes[1, 1].set_xticks(range(len(topic_counts)))
    axes[1, 1].set_xticklabels([t[:20] for t in topic_counts.index], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Number of Companies')
    axes[1, 1].set_title('Top 10 LDA Topics by Company Count', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gics_lda_alignment.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization as 'gics_lda_alignment.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def analyze_gics_lda_alignment(data_path, model_path):
    """
    Run complete GICS-LDA alignment analysis.
    
    Args:
        data_path: Path to company data pickle file
        model_path: Path to trained LDA model
    
    Returns:
        embeddings DataFrame with all analysis results
    """
    # Load data
    longD, ldaModel = load_data_and_model(data_path, model_path)
    
    # Infer topics
    embeddings = infer_topic_embeddings(longD, ldaModel)
    
    # Assign dominant topics
    embeddings, topicnames = assign_dominant_topics(embeddings, ldaModel)
    
    # Analyze counts
    gics_counts, topic_counts = analyze_company_counts(embeddings)
    
    # Cross-tabulations
    indGrpNames = get_gics_industry_names()
    gics_coherence = cross_tab_gics_to_lda(embeddings, indGrpNames)
    topic_purity = cross_tab_lda_to_gics(embeddings, topicnames)
    
    # Market cap analysis
    mcapGics, mcapTopic = analyze_market_cap(embeddings, topicnames)
    
    # Visualizations
    create_visualizations(embeddings, gics_coherence, topic_purity)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return embeddings


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage (requires actual data files)
    print("\nTo run this analysis, you need:")
    print("1. Company data file (e.g., 'longD_unit2.p')")
    print("2. Trained LDA model (e.g., 'ldaIndGrpCleanFinal.mdl')")
    print("\nExample usage:")
    print("  embeddings = analyze_gics_lda_alignment('longD_unit2.p', 'ldaIndGrpCleanFinal.mdl')")
    print("\nNote: This script is a template. Modify paths to match your data files.")
