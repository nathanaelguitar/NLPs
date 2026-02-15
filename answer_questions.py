import importlib
import Base
import pandas as pd
import numpy as np
import math 
from scipy.linalg import norm
import pickle 
import tomotopy as tp

importlib.reload(Base)
longD = Base.getDescriptions()
embeddings = Base.getEmbeddings()
topicnames = Base.gettopicnames()

def hellinger(p, q):
    """Hellinger distance between two discrete distributions."""
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
        list_of_squares.append(s)
    sosq = sum(list_of_squares)
    return sosq / math.sqrt(2)

def getCompetitive(tic, embeddings):
    p = np.array(embeddings[embeddings['ticker']==tic][["topic"+str(x) for x in range(0,24)]])[0]
    vec = []
    hDist = []
    for i, e in embeddings.iterrows():
        q = np.array(e[["topic"+str(x) for x in range(0,24)]])
        hd = hellinger(p, q)
        hDist.append([e.companyid,hd])
    hDist = pd.DataFrame(hDist, columns=['companyid','distance'])
    return hDist

def printComparables(tic, embeddings, longD, n=10):
    """Print top n comparable companies by Hellinger distance"""
    hDist = getCompetitive(tic, embeddings)
    comparables = embeddings.merge(hDist, on='companyid')
    comparables = comparables[['companyid','indGrp','topic','distance']]
    comparables = comparables.sort_values(by='distance')
    comparables = comparables.merge( longD[['companyid','compustat_name','ticker']], on='companyid' )
    print(f"\n===== TOP {n} COMPARABLES FOR {tic} =====")
    for i, c in comparables.head(n=n).iterrows():
        print(str(round(c['distance'],3)) + " " + str(c['compustat_name']) + " (" + str(c['ticker']) + ") " + str(c['topic']) + "    " + str(c['indGrp']) )

# Print comparables for all three companies
printComparables("SIIX", embeddings, longD, n=10)
printComparables("DSGN", embeddings, longD, n=10)
printComparables("FL", embeddings, longD, n=10)

# Word frequency analysis
def wordFrequencyTicker(tic, col):
    sdFreq = longD[longD['ticker']==tic][col].str.split(expand=True).stack().value_counts()
    sdFreq = pd.DataFrame(sdFreq).reset_index()
    sdFreq.columns = ['word','n']
    return sdFreq

wfSIIX = wordFrequencyTicker("SIIX", 'cleanDesc')
wfDSGN = wordFrequencyTicker("DSGN", 'cleanDesc')
wfFL = wordFrequencyTicker("FL", 'cleanDesc')

wfCompare = wfSIIX.merge(wfDSGN, how='outer', on='word').fillna(0)
wfCompare.columns = ['word','n_SIIX','n_DSGN']
wfCompare = wfCompare.merge(wfFL, how='outer', on='word').fillna(0)
wfCompare.columns = ['word','n_SIIX','n_DSGN','n_FL']

print("\n===== TOP 10 WORDS FOR SIIX =====")
print(wfCompare.sort_values(by='n_SIIX', ascending=False).head(n=10)[['word','n_SIIX']])

print("\n===== TOP 10 WORDS FOR DSGN =====")
print(wfCompare.sort_values(by='n_DSGN', ascending=False).head(n=10)[['word','n_DSGN']])

print("\n===== TOP 10 WORDS FOR FL =====")
print(wfCompare.sort_values(by='n_FL', ascending=False).head(n=10)[['word','n_FL']])

# Topic probabilities
def printEmbeddings(ticker):
    name = embeddings[embeddings['ticker']==ticker]['compustat_name']
    gig = embeddings[embeddings['ticker']==ticker]['indGrp']
    print(f"\n===== TOPIC PROBABILITIES FOR {ticker} =====")
    print(f"Company: {name.iloc[0]}")
    print(f"Industry Group: {gig.iloc[0]}")
    compEmbeddings = embeddings[embeddings['ticker']==ticker][["topic"+str(x) for x in range(0,24)]].T.reset_index()
    compEmbeddings.columns = ['topic','relevance']
    compEmbeddings = compEmbeddings.merge(topicnames, on='topic')
    compEmbeddings = compEmbeddings.sort_values(by='relevance', ascending=False)
    compEmbeddings = compEmbeddings[['relevance','topic_words']]
    for i,c in compEmbeddings.iterrows(): 
        print(str(round(c.relevance,3)) + " " + str(c.topic_words[0:5]))

printEmbeddings("SIIX")
printEmbeddings("DSGN")
printEmbeddings("FL")
