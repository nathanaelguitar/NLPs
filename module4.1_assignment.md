# Module 4.1 Assignment - Training and Evaluating a Doc2Vec Model

## Instructions

1. Train your first Doc2Vec model on the cleaned business descriptions from Unit 2, merging the resulting
   loadings to create vectors specific to each of the Middle 1000 companies.
2. Reduce the loadings of the Doc2Vec model from 150 dimensions (loadings) to 3 with both PCA and TSNE
   in order to visualize and compare/contrast the resulting vectors with Plotly.

## Deliverables

- Import the `cleanDesc` business descriptions for the Middle 1000 companies from Unit 2 then split the
  descriptions into sentences for utilization by the Doc2Vec algorithm.
- Merge in the GICS industry group names and print the business descriptions, sorted by market
  capitalization.
- List the top 5 companies and their associated GICS industry groups.  Do you agree with those GICS
  classifications?

- Train your first d2v model, merge the resulting loadings back into the `longD` dataframe, and print the
  loadings for the first vector.
  * List the first 10 loadings for the first vector.
  * How many loadings are in the first vector?
- Print the d2v loadings for Range Resources Corp. to verify that your merge of loadings into the df
  was effective.
  * List the first 10 loadings for the RRC vector.
  * Do those loadings match the loadings for the first vector?

- Run `sklearn.decomposition.PCA` on d2v vectors to reduce their dimensions (loadings) from 150 to 3.
  * Audit those loadings to verify reduction to 3 dimensions.
  * List PCA vector loadings of the top 10 companies by Market Cap.
- Run Plotly to build a 3D scatter plot of the d2v vectors.
  * Can you see some separation between GICS groups in the chart?
  * Is there some crowding around the center?
  * How effectively do PCA reduced Doc2Vec loadings capture the underlying meaning of business
    descriptions?

- Run `sklearn.manifold.TSNE` on d2v vectors to reduce their dimensions (loadings) from 150 to 3.
  * Audit those loadings to verify reduction to 3 dimensions.
  * List TSNE vector loadings of the top 10 companies by Market Cap.
- Run Plotly to build a 3D scatter plot of the d2v vectors.
  * Can you see some separation between GICS groups in the chart?
  * Is there some crowding around the center?
  * How effectively do TSNE reduced Doc2Vec loadings capture the underlying meaning of business
    descriptions?

- Which dimension-reducing algorithm is better for utilization in a machine learning algorithm?
- Which dimension-reducing algorithm is better for utilization in data visualization?

- Based upon the analysis of these data visualizations, describe how effective you think the Doc2Vec
  algorithm is at creating a numerical landscape that will allow us to separate out companies based upon
  their business activities.
  * Are there sectors/industries that the model seems better at separating?
  * Are there sectors/industries that the model seems worse at separating?
  * How effective would you say Doc2Vec was for the Middle 1000 companies compared to the Top 1000
    companies?
  * How effective would you say Doc2Vec is overall against the LDA model?
