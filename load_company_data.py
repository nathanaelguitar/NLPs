#!/usr/bin/env python
# coding: utf-8

"""
Load and prepare company data from CSV for Module 3.2 GICS alignment analysis

This script loads the course_base_data.csv and prepares it in the format
expected by Base.py and gics_alignment_analysis.py
"""

import pandas as pd
import numpy as np

# GICS Industry Group names (24 standard groups)
GICS_INDUSTRY_GROUPS = {
    1010: 'Energy',
    1510: 'Materials',
    2010: 'Capital Goods',
    2020: 'Commercial & Professional Services',
    2030: 'Transportation',
    2510: 'Automobiles & Components',
    2520: 'Consumer Durables & Apparel',
    2530: 'Consumer Services',
    2550: 'Retailing',
    3010: 'Food & Staples Retailing',
    3020: 'Food, Beverage & Tobacco',
    3030: 'Household & Personal Products',
    3510: 'Health Care Equipment & Services',
    3520: 'Pharmaceuticals, Biotechnology & Life Sciences',
    4010: 'Banks',
    4020: 'Diversified Financials',
    4030: 'Insurance',
    4510: 'Software & Services',
    4520: 'Technology Hardware & Equipment',
    4530: 'Semiconductors & Semiconductor Equipment',
    5010: 'Telecommunication Services',
    5020: 'Media & Entertainment',
    5510: 'Utilities',
    6010: 'Real Estate'
}


def load_company_data(csv_path='/Users/nathanaelguitar/Downloads/course_base_data.csv'):
    """
    Load company data from CSV and prepare for analysis.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with company information and GICS classifications
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Add GICS industry group (first 4 digits of gicsInd)
    df['gicsIndGrp'] = (df['gicsInd'] / 100).astype(int)
    
    # Add industry group names
    df['indGrp'] = df['gicsIndGrp'].map(GICS_INDUSTRY_GROUPS)
    
    # Sort by market cap (largest to smallest)
    df = df.sort_values(by='mktcap', ascending=False)
    df = df.reset_index(drop=True)
    
    print(f"\nLoaded {len(df)} companies")
    print(f"Market cap range: ${df['mktcap'].min():.2f}M - ${df['mktcap'].max():.2f}M")
    print(f"\nGICS Industry Groups represented: {df['gicsIndGrp'].nunique()}")
    
    return df


def analyze_gics_distribution(df):
    """
    Analyze the distribution of companies across GICS industry groups.
    
    Args:
        df: DataFrame with company data
    """
    print("\n" + "="*80)
    print("GICS INDUSTRY GROUP DISTRIBUTION")
    print("="*80)
    
    # Count by industry group
    ind_counts = df.groupby(['gicsIndGrp', 'indGrp']).agg({
        'companyid': 'count',
        'mktcap': ['sum', 'mean', 'median']
    }).round(2)
    
    ind_counts.columns = ['Count', 'Total_MktCap', 'Avg_MktCap', 'Median_MktCap']
    ind_counts = ind_counts.sort_values('Total_MktCap', ascending=False)
    
    print("\nTop 10 Industry Groups by Total Market Cap:")
    print(ind_counts.head(10))
    
    print("\n\nTop 10 Industry Groups by Company Count:")
    print(ind_counts.sort_values('Count', ascending=False).head(10))
    
    return ind_counts


def get_top_companies_by_industry(df, n=5):
    """
    Get top N companies by market cap for each industry group.
    
    Args:
        df: DataFrame with company data
        n: Number of top companies per industry
        
    Returns:
        DataFrame with top companies
    """
    print("\n" + "="*80)
    print(f"TOP {n} COMPANIES BY INDUSTRY GROUP")
    print("="*80)
    
    top_companies = []
    
    for ind_grp in sorted(df['gicsIndGrp'].unique()):
        ind_name = GICS_INDUSTRY_GROUPS.get(ind_grp, 'Unknown')
        ind_df = df[df['gicsIndGrp'] == ind_grp].head(n)
        
        print(f"\n{ind_name} ({ind_grp}):")
        for _, row in ind_df.iterrows():
            print(f"  {row['ticker']:6s} - {row['compustat_name']:40s} ${row['mktcap']:,.2f}M")
        
        top_companies.append(ind_df)
    
    return pd.concat(top_companies, ignore_index=True)


def simulate_topic_assignments(df, n_topics=24, random_state=42):
    """
    Simulate LDA topic assignments for testing (until we have actual model).
    
    In reality, these would come from inferring a trained LDA model.
    For now, we'll create plausible assignments based on industry.
    
    Args:
        df: DataFrame with company data
        n_topics: Number of topics
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with simulated topic probabilities and assignments
    """
    np.random.seed(random_state)
    
    print("\n" + "="*80)
    print("SIMULATING TOPIC ASSIGNMENTS (for testing)")
    print("="*80)
    print("Note: In production, use actual LDA model inference")
    
    # Create topic probability columns
    topic_cols = [f'topic{i}' for i in range(n_topics)]
    
    # Simulate topic probabilities
    # Companies in same industry will have higher probability for certain topics
    topic_probs = np.zeros((len(df), n_topics))
    
    for i, row in df.iterrows():
        # Base: random small probabilities
        probs = np.random.dirichlet(np.ones(n_topics) * 0.5)
        
        # Boost: assign higher probability to topics correlated with industry
        # Map industry group to preferred topics (just for simulation)
        ind_topic = hash(row['gicsIndGrp']) % n_topics
        probs[ind_topic] *= 5
        probs[ind_topic - 1] *= 2  # Adjacent topics also boosted
        
        # Normalize
        probs = probs / probs.sum()
        topic_probs[i] = probs
    
    # Add topic probabilities to dataframe
    for i, col in enumerate(topic_cols):
        df[col] = topic_probs[:, i]
    
    # Assign to dominant topic
    df['topic'] = df[topic_cols].idxmax(axis=1)
    df['topic_prob'] = df[topic_cols].max(axis=1)
    
    print(f"\nAssigned {len(df)} companies to {n_topics} topics")
    print(f"Average max probability: {df['topic_prob'].mean():.3f}")
    print(f"\nTopic distribution:")
    print(df['topic'].value_counts().sort_index())
    
    return df


def main():
    """Main execution function."""
    print("="*80)
    print("COMPANY DATA LOADER - MODULE 3.2")
    print("="*80)
    
    # Load data
    df = load_company_data()
    
    # Analyze GICS distribution
    ind_stats = analyze_gics_distribution(df)
    
    # Get top companies per industry
    top_companies = get_top_companies_by_industry(df, n=3)
    
    # Simulate topic assignments for testing
    df_with_topics = simulate_topic_assignments(df.copy(), n_topics=24)
    
    # Show example of bidirectional cross-tabulation
    print("\n" + "="*80)
    print("EXAMPLE: GICS → TOPIC CROSS-TABULATION")
    print("="*80)
    print("\nHow are GICS industry groups distributed across topics?")
    
    crosstab_gics_topic = pd.crosstab(
        df_with_topics['indGrp'],
        df_with_topics['topic'],
        margins=True,
        margins_name='Total'
    )
    print(crosstab_gics_topic)
    
    print("\n" + "="*80)
    print("EXAMPLE: TOPIC → GICS CROSS-TABULATION")
    print("="*80)
    print("\nHow are topics distributed across GICS industry groups?")
    
    crosstab_topic_gics = pd.crosstab(
        df_with_topics['topic'],
        df_with_topics['indGrp'],
        margins=True,
        margins_name='Total'
    )
    print(crosstab_topic_gics)
    
    print("\n" + "="*80)
    print("DATA READY FOR MODULE 3.2 ANALYSIS")
    print("="*80)
    print("\nNext steps:")
    print("1. Train an actual LDA model on company descriptions")
    print("2. Infer topic probabilities using the trained model")
    print("3. Run gics_alignment_analysis.py with real topic assignments")
    print("4. Compare alignment metrics across different models")
    
    return df, df_with_topics


if __name__ == "__main__":
    df, df_with_topics = main()
