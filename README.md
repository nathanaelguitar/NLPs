# NLPs

Tracking my NLP learning notes, experiments, and topic-modeling practice.

## Overview

This repository documents my journey learning Natural Language Processing, with a focus on:
- Text preprocessing and cleaning
- Frequency analysis and stopword filtering
- Topic modeling using LDA (Latent Dirichlet Allocation)
- Hyperparameter tuning and model evaluation
- Interactive visualization with pyLDAvis

**Key Learning:** *"Clean text determines meaningful topics."* Most NLP performance comes from preprocessing, not fancy algorithms.

---

## Module 3.2: LDA Topic Modeling and GICS Industry Alignment

### Overview

This module evaluates how well LDA topic groupings align with GICS (Global Industry Classification Standard) industry groups using the Middle 1000 companies by market capitalization. The trained model has 24 topics, and the core workflow involves: inferring topic probabilities for each company, assigning each to its highest-probability topic, then cross-tabulating LDA topics against GICS groups bidirectionally.

### Key Workflow Steps

**1. Infer Topic Probabilities (Embeddings)**
- Convert each company's cleaned description into a 24-element probability vector
- Each element represents alignment strength with a specific topic
- Output: Soft assignments showing how companies distribute across topics

**2. Hard Assignment to Dominant Topic**
- Use `idxmax()` to assign each company to its highest-probability topic
- Creates winner-take-all classifications while preserving soft probabilities
- Example: Company with `topic12=0.88` assigned to topic12

**3. Topic Naming from Top Words**
- Extract top 5-10 words per topic using `get_topic_words()`
- Create human-readable names (e.g., `loan_bank_banking_deposit_lending_`)
- Topic numbers are arbitrary; names identify actual content

**4. Cross-Tabulation Analysis (Two Directions)**

**GICS → LDA:** How thematically coherent is each GICS group?
- **High consistency**: Utilities (96.2%), Insurance (95.0%), Banks (91.8%)
- **Moderate consistency**: Semiconductors (81.2%), Autos (71.4%)
- **Low consistency**: Consumer Services (33.3%), Software (35.4%)

**LDA → GICS:** Does each topic correspond to single or multiple industries?
- **Pure topics**: Some pharma topics are 100% Pharmaceuticals
- **Focused topics**: Energy topic is 88.9% Energy sector
- **Mixed topics**: Semiconductor topic spans Tech Hardware (45.2%) + Semiconductors (31.0%)

**5. Market Capitalization Analysis**
- **GICS market cap**: Hard assignment (each company in one group)
- **LDA market cap**: Soft assignment weighted by topic probabilities
- Reveals that companies straddle multiple themes economically

### Key Findings

**1. Alignment Patterns by Industry Type:**
- **Strong alignment**: Specialized industries (Banks, Utilities, Insurance) have distinctive vocabulary
- **Weak alignment**: Diversified services (Software, Consumer Services) scatter across topics
- **Sub-structure splitting**: Pharma fragments into clinical trials, oncology, testing—more granular than GICS

**2. Cross-Industry Clusters Are Real:**
- Topic12 (semiconductor/component/device) legitimately spans Tech Hardware + Semiconductors
- Reflects genuine functional similarity in business activities and language

**3. Sample Size Effects:**
- Small groups (Food & Staples: 8 companies, Telecom: 10) produce less stable assignments
- Large groups (Capital Goods: 106, Software: 79) show clearer patterns

**4. What This Reveals:**
- **LDA captures vocabulary, not classification** - groups by self-description, not official category
- **Topic splitting reveals hidden structure** - finds meaningful sub-segments within GICS groups
- **Cross-GICS topics are informative** - show where industry boundaries blur operationally

### Practical Implications

**When LDA outperforms GICS:**
- Discovering sub-industries within broad categories
- Finding cross-industry functional clusters
- Identifying companies that don't fit their GICS classification

**When GICS outperforms LDA:**
- Need for standardization across studies
- Regulatory and investor communication
- Historical comparability

**Optimal approach:** Use LDA for discovery and pattern detection, then map insights to GICS for practical application and communication.

---

## Key Insights: LDA vs Traditional Classification (GICS)

LDA provides a **bottom-up, text-driven classification** that often cuts across GICS boundaries, revealing economically meaningful clusters that fixed taxonomies miss. 

### When LDA Excels:
- **Finer granularity within sectors** (e.g., separating upstream energy, pipelines, and utilities)
- **Multi-topic memberships** capture hybrid business models that GICS forces into a single bucket
- **Exploratory analysis** and detecting latent structure in text data
- **Adaptability** to corpus-specific patterns

### LDA Limitations:
- Some assignments reflect **disclosure style rather than true business activity**, limiting firm-level relevance
- **Sensitive to reporting bias**, boilerplate language, and corpus composition
- **Sample-dependent** results are less stable over time
- **Not directly comparable** across studies (no standardization)

### When GICS Excels:
- **Standardization** that investors and regulators rely on
- **Consistency** for communication and benchmarking
- **Human judgment** embedded in classifications
- **Cross-study comparability**

### GICS Limitations:
- Can **lag structural change** due to historical definitions
- **Forces diversified firms** into single categories
- May **miss emerging industry patterns**

**Bottom Line:** LDA is superior for exploratory analysis and revealing hidden structure. GICS remains essential for consistency and standardized communication. The optimal approach often combines both: use LDA to discover patterns, then map insights to GICS for practical application.

---

## Key Insights: Comparable Companies Analysis with Hellinger Distance

A comparable companies list generated from **LDA topics and Hellinger distance** stops being effective when similarity ranking begins to reflect **generic language rather than shared core operations**.

### When the Similarity Breaks Down:

**Early neighbors (high quality):**
- Share dominant topics tied to main revenue drivers
- Reflect true economic activity overlap
- Useful for valuation and peer analysis

**Later neighbors (degrading quality):**
- Distances increase gradually
- Topic overlap becomes superficial
- Common business terms replace industry-specific vocabulary
- Matching on **communication style** instead of **economic activity**

### Determining the Cutoff:

**Quantitative signals:**
1. **Elbow in the distance curve** - where distances start increasing rapidly
2. **Monitor marginal change** - stop when Hellinger distance between consecutive neighbors shows large jumps
3. **Topic probability alignment** - verify top topics still match the firm's primary segment

**Qualitative validation:**
1. **Scan company descriptions** to confirm they truly compete in the same space
2. **Check industry-specific terminology** vs generic business language
3. **Verify operational overlap** beyond textual similarity

### Practical Implications:

**Above the threshold (reliable):**
- Use for valuation comparables
- Trust for peer analysis
- Reflects true competitive relationships

**Below the threshold (unreliable):**
- Mixes true comparables with textually similar firms
- Less relevant for financial analysis
- May mislead if used without validation

**Best Practice:** Always combine quantitative distance metrics with qualitative review of company descriptions to ensure comparable lists reflect genuine economic similarity, not just linguistic patterns.

## Projects

### 1. `nlp_business_learning.ipynb` 
Comprehensive Jupyter notebook covering:
- Module 1.2: Data preparation & source consistency
- Module 1.3: How LDA topic modeling works
- Module 1.4: Stopwords & frequency analysis
- Module 1.5: Topic modeling experiments
- Module 2.1: Regex text cleaning
- Module 3.2: LDA and GICS industry alignment analysis
- **Validation Module**: Quantitative topic validation with term-topic assignments and separation metrics

Uses the 20 Newsgroups dataset to demonstrate the complete NLP pipeline including validation.

### 2. `topic_modeling_newsgroups.py`
Python script version of the topic modeling pipeline with automated hyperparameter tuning.

### 3. `lda_validation.py`
Standalone validation utilities for LDA models:
- **Term-topic assignment**: Find which topic maximizes P(word|topic)
- **Topic separation**: Quantify overlap using cosine similarity
- **Term weight analysis**: Compare related terms across topics
- **Validation reports**: Comprehensive model quality assessment

Import and use:
```python
from lda_validation import *
validate_model(model, coherence_score=0.45, 
               terms_to_check=["computer", "medical", "space"],
               output_prefix="my_analysis")
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
python3 imports_check.py
```

Recommended Python version: `3.12` (both `gensim` and `tomotopy` are currently problematic on `3.14` in this environment).

## Libraries in use

- `tomotopy`
- `gensim`
- `matplotlib`
- `wordcloud`
- `pyLDAvis`
- `collections`
- `pickle`
- `time`

## Notes

- `import Base` is not a standard Python package. If needed, add a local `Base.py` module or replace it with the intended package name.
