#!/usr/bin/env python
# coding: utf-8

"""
Base.py - Helper module for loading company data and LDA model

This module provides convenience functions for:
- Loading company descriptions with GICS classifications
- Inferring topic embeddings from trained LDA model
- Getting topic names from model

Prerequisites:
- longD_unit2.p: Pickled DataFrame with company data
- ldaIndGrpCleanFinal.mdl: Trained 24-topic LDA model

Note: This is adapted from the course-provided Base.py module.
"""

import os
import pickle
import tomotopy as tp
import pandas as pd


def getDescriptions():
    """
    Load and prepare company descriptions DataFrame.
    
    Returns:
        DataFrame with columns:
        - companyid, gvkey, compustat_name, ticker, mktcap
        - gics, gicsInd, gicsIndGrp, gicsSubind
        - cleanDesc (preprocessed text)
        - nWords (word count)
    """
    try:
        # Try to read with pandas first
        longD = pd.read_pickle("longD_unit2.p")
    except:
        # Fallback: load with pickle
        longD = pickle.load(open("longD_unit2.p", "rb"), encoding='latin1')
    longD = longD.sort_values(by='mktcap', ascending=False)  # Order largest to smallest
    longD = longD.reset_index(drop=True)  # Reset index in market cap order
    longD = longD[longD['gics'] > 0]  # Filter to companies with GICS codes
    longD['nWords'] = longD['cleanDesc'].str.split().apply(len)
    longD = longD[longD['nWords'] > 5]  # Keep descriptions with >5 words
    longD['gicsIndGrp'] = (longD['gicsInd'] / 100).astype(int)
    
    return longD


def getEmbeddings():
    """
    Load company descriptions, infer topic probabilities, and assign to topics.
    
    Returns:
        DataFrame with:
        - topic0 through topic23: Topic probabilities for each company
        - Company metadata (companyid, gvkey, name, ticker, mktcap, GICS codes)
        - indGrp: GICS industry group name
        - topic: Assigned dominant topic (e.g., "topic12")
        - topic_name: Human-readable topic name from top words
        - topic_words: List of top 10 words for the topic
    """
    # Load data
    try:
        longD = pd.read_pickle("longD_unit2.p")
    except:
        longD = pickle.load(open("longD_unit2.p", "rb"), encoding='latin1')
    longD = longD.sort_values(by='mktcap', ascending=False)
    longD = longD.reset_index(drop=True)
    longD = longD[longD['gics'] > 0]
    longD['nWords'] = longD['cleanDesc'].str.split().apply(len)
    longD = longD[longD['nWords'] > 5]
    longD['gicsIndGrp'] = (longD['gicsInd'] / 100).astype(int)
    
    # Load model
    ldaIndGrpClean = tp.LDAModel.load("ldaIndGrpCleanFinal.mdl")
    
    # Infer topic probabilities
    companyId = [item for sublist in longD[['companyid']].values.tolist() for item in sublist]
    cleanDesc = [doc.split() for doc in longD['cleanDesc'].astype(str)]
    embeddings = [ldaIndGrpClean.infer(ldaIndGrpClean.make_doc(desc)) for desc in cleanDesc]
    embeddings = pd.DataFrame([x[0] for x in embeddings])
    embeddings.columns = ["topic" + str(x) for x in range(0, 24)]
    embeddings.index = companyId
    embeddings = embeddings.reset_index()
    embeddings = embeddings.rename(columns={'index': 'companyid'})
    embeddings = embeddings.merge(
        longD[['companyid', 'gvkey', 'compustat_name', 'ticker', 'mktcap',
               'gics', 'gicsIndGrp', 'gicsInd', 'gicsSubind']],
        on='companyid'
    )
    
    # Add GICS industry group names
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
    embeddings = embeddings.merge(indGrpNames, on='gicsIndGrp')
    
    # Assign to dominant topic (hard assignment)
    maxtopic = embeddings[["topic" + str(x) for x in range(0, 24)]].idxmax(axis=1)
    embeddings = embeddings.merge(maxtopic.rename('topic'), left_index=True, right_index=True)
    
    # Create topic names from top words
    topicnames = []
    for k in range(0, 24):
        nm = ""
        for n in ldaIndGrpClean.get_topic_words(k, top_n=5):
            nm = nm + n[0] + "_"
        topicnames.append([
            "topic" + str(k), k, nm,
            [x[0] for x in ldaIndGrpClean.get_topic_words(k, top_n=10)]
        ])
    topicnames = pd.DataFrame(topicnames)
    topicnames.columns = ['topic', 'topicn', 'topic_name', 'topic_words']
    embeddings = embeddings.merge(topicnames, on='topic')
    
    return embeddings


def gettopicnames():
    """
    Load LDA model and extract topic names from top words.
    
    Returns:
        DataFrame with columns:
        - topic: Topic identifier (e.g., "topic0")
        - topicn: Topic number (0-23)
        - topic_name: Concatenated top 5 words
        - topic_words: List of top 10 words
    """
    ldaIndGrpClean = tp.LDAModel.load("ldaIndGrpCleanFinal.mdl")
    
    topicnames = []
    for k in range(0, 24):
        nm = ""
        for n in ldaIndGrpClean.get_topic_words(k, top_n=5):
            nm = nm + n[0] + "_"
        topicnames.append([
            "topic" + str(k), k, nm,
            [x[0] for x in ldaIndGrpClean.get_topic_words(k, top_n=10)]
        ])
    topicnames = pd.DataFrame(topicnames)
    topicnames.columns = ['topic', 'topicn', 'topic_name', 'topic_words']
    
    return topicnames


def getldaIndGrpClean():
    """
    Load the trained LDA model.
    
    Returns:
        Tomotopy LDAModel object
    """
    return tp.LDAModel.load("ldaIndGrpCleanFinal.mdl")
