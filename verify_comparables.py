import pickle
import pandas as pd
import numpy as np
import math

# Load the embeddings
embeddings = pickle.load(open("embeddings_unit3.p", "rb"))

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

# Load longD for company names
import Base
longD = Base.getDescriptions()

print("=" * 60)
print("DSGN TOP 10 COMPARABLES")
print("=" * 60)
hDist = getCompetitive("DSGN", embeddings)
comparables = embeddings.merge(hDist, on='companyid')
comparables = comparables[['companyid','indGrp','topic','distance']]
comparables = comparables.sort_values(by='distance')
comparables = comparables.merge(longD[['companyid','compustat_name','ticker']], on='companyid')
for i, c in comparables.head(n=10).iterrows():
    print(f"{c['distance']:.3f} {c['compustat_name']} ({c['ticker']})")

print("\n" + "=" * 60)
print("FL TOP 10 COMPARABLES")
print("=" * 60)
hDist = getCompetitive("FL", embeddings)
comparables = embeddings.merge(hDist, on='companyid')
comparables = comparables[['companyid','indGrp','topic','distance']]
comparables = comparables.sort_values(by='distance')
comparables = comparables.merge(longD[['companyid','compustat_name','ticker']], on='companyid')
for i, c in comparables.head(n=10).iterrows():
    print(f"{c['distance']:.3f} {c['compustat_name']} ({c['ticker']})")
