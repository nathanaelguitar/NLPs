import os 
import pickle 
import pandas as pd
import itertools
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import plotly
import plotly.express as px
from sklearn.manifold import TSNE 
import warnings

DOC2VEC_MODEL_PATH = "/Users/nathanaelguitar/Downloads/d2v_model.p"


def load_or_train_doc2vec(docs, model_path=DOC2VEC_MODEL_PATH):
    if os.path.exists(model_path):
        print(f"Loading Doc2Vec model from {model_path}")
        return Doc2Vec.load(model_path)

    print("Training a new Doc2Vec model")
    d2v = Doc2Vec(vector_size=150       # Number of loadings the doc2vec algorithm will generate
                  , min_count=5         # Ignores all words with total frequency lower than this
                  , epochs=25           # Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.
                  , window=8            # The maximum distance between the current and predicted word within a sentence.
                  , sample=1e-5)        # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is
    d2v.workers = 8
    d2v.build_vocab(docs)
    d2v.train(docs, total_examples=d2v.corpus_count, epochs=d2v.epochs)
    return d2v


def get_docvec_store(model):
    return getattr(model, "dv", None) or model.__dict__.get("docvecs")


def get_doc_vector(model, key):
    docvecs = get_docvec_store(model)
    if docvecs is None:
        raise AttributeError("Doc2Vec model does not expose document vectors.")

    if hasattr(docvecs, "key_to_index"):
        return docvecs[key]

    if hasattr(docvecs, "offset2doctag") and hasattr(docvecs, "vectors_docs"):
        doc_index = {tag: i for i, tag in enumerate(docvecs.offset2doctag)}
        return docvecs.vectors_docs[doc_index[key]]

    raise TypeError("Unsupported document vector store.")


def get_doc_vector_size(model):
    docvecs = get_docvec_store(model)
    if docvecs is None:
        raise AttributeError("Doc2Vec model does not expose document vectors.")

    if hasattr(docvecs, "vector_size"):
        return docvecs.vector_size

    first_key = next(iter(docvecs.doctags))
    return len(docvecs[first_key])


def load_longd(path="longD_unit2.p"):
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def safe_show(fig):
    try:
        fig.show()
    except Exception as exc:
        print(f"Skipping interactive plot display: {exc}")


longD = load_longd("longD_unit2.p")
longD = longD[longD['gics']>0]                             # Make sure it has an industry 
longD = longD[longD['mktcap']>0]                           # Make sure it has a market cap
longD['nWords'] = longD['cleanDesc'].str.split().apply(len)
longD = longD[longD['nWords']>5]                           # Make sure it has more than 5 words
longD['gicsIndGrp'] = (longD['gicsInd']/100).astype(int)   # Add Industry Group of 24 as a column 

def senSplit(v):
    result = []
    for sen in v:
        result.append(sen.split(' '))
    return result

longD['cleanSentList'] = longD['cleanSent'].apply(senSplit)

indGrpNames = pd.DataFrame([[1010,'Energy']
,[1510,'Materials']
,[2010,'Capital Goods']
,[2020,'Commercial & Professional Services']
,[2030,'Transportation']
,[2510,'Automobiles & Components']
,[2520,'Consumer Durables & Apparel']
,[2530,'Consumer Services']
,[2550,'Retailing']
,[3010,'Food & Staples Retailing']
,[3020,'Food, Beverage & Tobacco']
,[3030,'Household & Personal Products']
,[3510,'Health Care Equipment & Services']
,[3520,'Pharmaceuticals, Biotechnology & Life Sciences']
,[4010,'Banks']
,[4020,'Diversified Financials']
,[4030,'Insurance']
,[4510,'Software & Services']
,[4520,'Technology Hardware & Equipment']
,[4530,'Semiconductors & Semiconductor Equipment']
,[5010,'Telecommunication Services']
,[5020,'Media & Entertainment']
,[5510,'Utilities']
,[6010,'Real Estate']])
indGrpNames.columns = ['gicsIndGrp','indGrp']
longD['gicsIndGrp'] = (longD['gicsInd']/100).astype(int)   # Add Industry Group of 24 as a column 
longD = longD.merge(indGrpNames, on='gicsIndGrp')

longD = longD.sort_values(by='mktcap', ascending=False)    # Order largest to smallest 
longD = longD.reset_index(drop=True)                       # Make sure the index is reset in the order of market cap 
print(longD)

for i, d in longD.iterrows(): 
    print(str(d.ticker) + " " + d.compustat_name)

def collapse_nested(l):
    return list(itertools.chain.from_iterable(l))

docs = [TaggedDocument(collapse_nested(longD['cleanSentList'].iloc[i]), 
                       [longD['cleanDesc'].iloc[i]]) for i in range(len(longD))]

d2v = load_or_train_doc2vec(docs)

print("Completed")

vec = []
for f in longD['cleanDesc']:
    vec.append(get_doc_vector(d2v, f))

scaler = StandardScaler()
scaled_vec = scaler.fit_transform(vec)

print("# Vectors: " + str(len(vec)))
print("First Vector")
print(scaled_vec[0])

warnings.filterwarnings('ignore')

d2vDf = longD.copy()
d2vDf['logmcap'] = np.log(d2vDf['mktcap'])

for i in range(get_doc_vector_size(d2v)):
   d2vDf['d2v'+str(i)] = 0.0

d2vDf.iloc[:,17:] = scaled_vec

print(d2vDf[d2vDf.ticker=="RRC"])

print(type(vec))
pca = PCA(n_components=3)
pca_result = pca.fit_transform(vec)
print(pca_result)

d2vDf['pcaX'] = pca_result[:,0]
d2vDf['pcaY'] = pca_result[:,1]
d2vDf['pcaZ'] = pca_result[:,2]

print("Check Company Vectors - PCA")
for index, row in d2vDf.head(n=10).iterrows():
  print( row['compustat_name'] +  ' ' + str(row['mktcap']) +' X:'+ str(row['pcaX']) + ' Y:'+ str(row['pcaY']) + ' Z:'+ str(row['pcaZ']))

fig1 = px.scatter_3d(d2vDf, x="pcaX", y="pcaY", z="pcaZ"
                 , color="gicsInd"
                 , hover_name="compustat_name"
                 , size="logmcap"
                 , size_max=25
                 , height=800, width=800
                     , opacity = 1.0
                    )
fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig1.write_html("d2v_pca.html")
safe_show(fig1)

vec = np.array(vec)
model = TSNE(n_components=3, random_state=0)
tsne_data = model.fit_transform(vec)

print(tsne_data)

d2vDf['tsneX'] = tsne_data[:,0]
d2vDf['tsneY'] = tsne_data[:,1]
d2vDf['tsneZ'] = tsne_data[:,2]

print("Check Company Vectors - TSNE")
for index, row in d2vDf.head(n=10).iterrows():
  print( row['compustat_name'] +  ' ' + str(row['mktcap']) +' X:'+ str(row['tsneX']) + ' Y:'+ str(row['tsneY']) + ' Z:'+ str(row['tsneZ']))

fig2 = px.scatter_3d(d2vDf, x="tsneX", y="tsneY", z="tsneZ"
                 , color="gicsInd"
                 , hover_name="compustat_name"
                 , size="logmcap"
                 , size_max=25
                 , height=800, width=800
                     , opacity = 1.0
                    )
fig2.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig2.write_html("d2v_tsne.html")
safe_show(fig2)
