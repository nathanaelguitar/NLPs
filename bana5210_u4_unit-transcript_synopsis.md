# BANA 5210 Unit 4 Synopsis

## Overview

Unit 4 introduces **Doc2Vec** as an alternative to **LDA** for NLP-driven financial analysis. The unit frames model selection as a practical machine learning decision: LDA is strong for topic creation from word frequencies, while Doc2Vec is useful when you need continuous semantic vectors that can support similarity search, visualization, and clustering.

The unit is organized around three goals:

1. Train and evaluate a Doc2Vec model.
2. Use Doc2Vec loadings to find comparable companies and theme-aligned firms.
3. Apply clustering and dimension reduction to Doc2Vec outputs.

## Core Idea: What Doc2Vec Adds

Unlike LDA, Doc2Vec does not generate topic probabilities directly from word frequencies. Instead, it converts documents into numerical vectors that preserve semantic context and word relationships. In the course framing, this makes Doc2Vec especially useful for:

- comparing companies by semantic similarity
- identifying firms aligned with a theme or concept
- constructing custom vectors for investment themes
- creating inputs for downstream algorithms such as clustering

The tradeoff is interpretability. Doc2Vec vectors are powerful, but the individual loadings are not directly human-readable in the way LDA topics are.

## Module 4.1: Train and Evaluate a Doc2Vec Model

The first module focuses on preparing cleaned business descriptions for Doc2Vec and producing company-level semantic vectors.

Key steps include:

- loading previously cleaned company descriptions
- filtering observations for usable GICS and market cap data
- preserving sentence order, because Doc2Vec uses context and sequence
- converting each company description into `TaggedDocument` objects
- training a `gensim` Doc2Vec model with selected hyperparameters such as `vector_size`, `window`, `epochs`, and `min_count`
- merging the resulting vectors back into the dataframe

The module then introduces **dimension reduction** to make high-dimensional outputs easier to inspect visually. Two methods are emphasized:

- **PCA** for linear reduction and easier interpretability
- **t-SNE** for visual pattern discovery in nonlinear structure

The evaluation lens is practical rather than purely statistical: do the resulting visual groupings and company relationships make sense relative to sectors and known business activity?

## Module 4.2: Use Doc2Vec to Find Comparable Companies

The second module applies Doc2Vec vectors to a core finance use case: identifying comparable companies. Instead of grouping firms only through preset taxonomies like GICS, the course uses distance metrics on semantic vectors to find companies that describe similar activities.

This module covers:

- distance-based comparison between company vectors
- comparing Doc2Vec-based peers against GICS and LDA outputs
- finding similar words through semantic vectors
- building custom theme vectors, including examples tied to ETFs such as **BOTZ** and **ARKG**
- using semantic alignment for thematic portfolio construction

The major takeaway is that Doc2Vec can be stronger than rigid industry classifications when the goal is to identify nuanced business similarity. It is especially valuable when firms operate in overlapping themes that are not captured cleanly by standard taxonomies.

## Module 4.3: Apply Doc2Vec Through Clustering

The third module addresses a limitation of Doc2Vec: it gives vectors, not groups. To turn those vectors into topic-like buckets for portfolio analysis and risk control, the unit applies clustering algorithms.

The main workflow is:

- feed Doc2Vec loadings into **KMeans**
- compare resulting clusters with **GICS industry groups**
- evaluate whether clusters are coherent and usable
- test **PCA-reduced** Doc2Vec vectors as engineered inputs for clustering

The results are mixed. Some clusters align well with GICS and produce intuitive peer groups, but many are noisy or overly broad. Several findings stand out:

- KMeans plus Doc2Vec can identify some coherent pockets, such as narrowly defined mining or banking groups.
- Other clusters are too heterogeneous to be useful.
- Compared with LDA, Doc2Vec plus KMeans is generally less clean for producing interpretable groupings.
- PCA reduction worsens clustering performance in this unit’s examples by collapsing too much useful information.

## Main Conclusions

- **Doc2Vec is useful when similarity matters more than explicit topic labels.**
- **LDA remains stronger for clean topic grouping and interpretability.**
- **Doc2Vec is well suited to comparable-company analysis and thematic investing.**
- **Clustering Doc2Vec outputs is possible, but results depend heavily on feature engineering, algorithm choice, and tuning.**
- **Model selection should follow the business question rather than a one-model-fits-all mindset.**

## Unit 4 Glossary

- **Doc2Vec**: an algorithm that creates a numeric representation of a document by vectorizing semantic word space into document loadings.
- **K-means algorithm**: a clustering method that groups observations by shared characteristics using distances between vectors.
- **Principal Component Analysis (PCA)**: a dimension reduction method that compresses many variables into fewer components with minimal information loss.
- **Thematic portfolio**: an investment portfolio built around companies tied to a common idea or trend, helping spread risk across that theme.
- **T-distributed Stochastic Neighbor Embedding (t-SNE)**: a dimension reduction method used to make high-dimensional data easier to visualize, usually with more information loss than PCA but often better local visual separation.

## Practical Finance Takeaways

From an investment perspective, the unit’s message is that NLP models can support two different but related tasks:

- **peer identification**: finding firms that operate similarly even if standard classifications miss the connection
- **portfolio construction and risk control**: creating groups or themes that can guide exposure management

Doc2Vec appears more compelling for the first task than the second in this unit. For grouping and portfolio-level categorization, LDA still appears more robust with less engineering.

## Final Takeaway

Unit 4 broadens the course from topic modeling into a more general machine learning workflow. The emphasis is not just on using Doc2Vec, but on understanding how preprocessing, hyperparameters, visualization, clustering, and ensembling all affect the quality of an NLP solution. The broader lesson is that useful financial NLP depends on matching the algorithm to the actual investment problem being solved.
