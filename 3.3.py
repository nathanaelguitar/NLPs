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

subInd = pd.DataFrame( [[10101010,'Oil & Gas Drilling']
,[10101020,'Oil & Gas Equipment & Services']
,[10102010,'Integrated Oil & Gas']
,[10102020,'Oil & Gas Exploration & Production']
,[10102030,'Oil & Gas Refining & Marketing']
,[10102040,'Oil & Gas Storage & Transportation']
,[10102050,'Coal & Consumable Fuels']
,[15101010,'Commodity Chemicals']
,[15101020,'Diversified Chemicals']
,[15101030,'Fertilizers & Agricultural Chemicals']
,[15101040,'Industrial Gases']
,[15101050,'Specialty Chemicals']
,[15102010,'Construction Materials']
,[15103010,'Metal & Glass Containers']
,[15103020,'Paper Packaging']
,[15104010,'Aluminum']
,[15104020,'Diversified Metals & Mining']
,[15104025,'Copper']
,[15104030,'Gold']
,[15104040,'Precious Metals & Minerals']
,[15104045,'Silver']
,[15104050,'Steel']
,[15105010,'Forest Products']
,[15105020,'Paper Products']
,[20101010,'Aerospace & Defense']
,[20102010,'Building Products']
,[20103010,'Construction & Engineering']
,[20104010,'Electrical Components & Equipment']
,[20104020,'Heavy Electrical Equipment']
,[20105010,'Industrial Conglomerates']
,[20106010,'Construction Machinery & Heavy Trucks']
,[20106015,'Agricultural & Farm Machinery']
,[20106020,'Industrial Machinery']
,[20107010,'Trading Companies & Distributors']
,[20201010,'Commercial Printing']
,[20201050,'Environmental & Facilities Services']
,[20201060,'Office Services & Supplies']
,[20201070,'Diversified Support Services']
,[20201080,'Security & Alarm Services']
,[20202010,'Human Resource & Employment Services']
,[20202020,'Research & Consulting Services']
,[20301010,'Air Freight & Logistics']
,[20302010,'Airlines']
,[20303010,'Marine']
,[20304010,'Railroads']
,[20304020,'Trucking']
,[20305010,'Airport Services']
,[20305020,'Highways & Railtracks']
,[20305030,'Marine Ports & Services']
,[25101010,'Auto Parts & Equipment']
,[25101020,'Tires & Rubber']
,[25102010,'Automobile Manufacturers']
,[25102020,'Motorcycle Manufacturers']
,[25201010,'Consumer Electronics']
,[25201020,'Home Furnishings']
,[25201030,'Homebuilding']
,[25201040,'Household Appliances']
,[25201050,'Housewares & Specialties']
,[25202010,'Leisure Products']
,[25203010,'Apparel, Accessories & Luxury Goods']
,[25203020,'Footwear']
,[25203030,'Textiles']
,[25301010,'Casinos & Gaming']
,[25301020,'Hotels, Resorts & Cruise Lines']
,[25301030,'Leisure Facilities']
,[25301040,'Restaurants']
,[25302010,'Education Services']
,[25302020,'Specialized Consumer Services']
,[25501010,'Distributors']
,[25502020,'Internet & Direct Marketing Retail']
,[25503010,'Department Stores']
,[25503020,'General Merchandise Stores']
,[25504010,'Apparel Retail']
,[25504020,'Computer & Electronics Retail']
,[25504030,'Home Improvement Retail']
,[25504040,'Specialty Stores']
,[25504050,'Automotive Retail']
,[25504060,'Homefurnishing Retail']
,[30101010,'Drug Retail']
,[30101020,'Food Distributors']
,[30101030,'Food Retail']
,[30101040,'Hypermarkets & Super Centers']
,[30201010,'Brewers']
,[30201020,'Distillers & Vintners']
,[30201030,'Soft Drinks']
,[30202010,'Agricultural Products']
,[30202030,'Packaged Foods & Meats']
,[30203010,'Tobacco']
,[30301010,'Household Products']
,[30302010,'Personal Products']
,[35101010,'Health Care Equipment']
,[35101020,'Health Care Supplies']
,[35102010,'Health Care Distributors']
,[35102015,'Health Care Services']
,[35102020,'Health Care Facilities']
,[35102030,'Managed Health Care']
,[35103010,'Health Care Technology']
,[35201010,'Biotechnology']
,[35202010,'Pharmaceuticals']
,[35203010,'Life Sciences Tools & Services']
,[40101010,'Diversified Banks']
,[40101015,'Regional Banks']
,[40102010,'Thrifts & Mortgage Finance']
,[40201020,'Other Diversified Financial Services']
,[40201030,'Multi-Sector Holdings']
,[40201040,'Specialized Finance']
,[40202010,'Consumer Finance']
,[40203010,'Asset Management & Custody Banks']
,[40203020,'Investment Banking & Brokerage']
,[40203030,'Diversified Capital Markets']
,[40203040,'Financial Exchanges & Data']
,[40204010,'Mortgage REITs']
,[40301010,'Insurance Brokers']
,[40301020,'Life & Health Insurance']
,[40301030,'Multi-line Insurance']
,[40301040,'Property & Casualty Insurance']
,[40301050,'Reinsurance']
,[45102010,'IT Consulting & Other Services']
,[45102020,'Data Processing & Outsourced Services']
,[45102030,'Internet Services & Infrastructure']
,[45103010,'Application Software']
,[45103020,'Systems Software']
,[45201020,'Communications Equipment']
,[45202030,'Technology Hardware, Storage & Peripherals']
,[45203010,'Electronic Equipment & Instruments']
,[45203015,'Electronic Components']
,[45203020,'Electronic Manufacturing Services']
,[45203030,'Technology Distributors']
,[45301010,'Semiconductor Equipment']
,[45301020,'Semiconductors']
,[50101010,'Alternative Carriers']
,[50101020,'Integrated Telecommunication Services']
,[50102010,'Wireless Telecommunication Services']
,[50201010,'Advertising']
,[50201020,'Broadcasting']
,[50201030,'Cable & Satellite']
,[50201040,'Publishing']
,[50202010,'Movies & Entertainment']
,[50202020,'Interactive Home Entertainment']
,[50203010,'Interactive Media & Services']
,[55101010,'Electric Utilities']
,[55102010,'Gas Utilities']
,[55103010,'Multi-Utilities']
,[55104010,'Water Utilities']
,[55105010,'Independent Power Producers & Energy Traders']
,[55105020,'Renewable Electricity']
,[60101010,'Diversified REITs']
,[60101020,'Industrial REITs']
,[60101030,'Hotel & Resort REITs']
,[60101040,'Office REITs']
,[60101050,'Health Care REITs']
,[60101060,'Residential REITs']
,[60101070,'Retail REITs']
,[60101080,'Specialized REITs']
,[60102010,'Diversified Real Estate Activities']
,[60102020,'Real Estate Operating Companies']
,[60102030,'Real Estate Development']
,[60102040,'Real Estate Services']] ) 

subInd.columns = ['gicsSubind','subInd']

nSubInd = pd.DataFrame( longD.groupby(['gicsSubind']).size() ).reset_index()
nSubInd.columns = ['gicsSubind','n']
subInd = subInd.merge(nSubInd, on='gicsSubind')


for i, s in subInd.iterrows():
    print(str(s.gicsSubind) + " " + str(s.n) + "   " + s.subInd)

print(longD[longD.gicsSubind==30201030])

print(longD[longD.gicsSubind==35201010])

tic = "SIIX"

name = embeddings[embeddings['ticker']==tic]['compustat_name']
gig = embeddings[embeddings['ticker']==tic]['indGrp']
print("###### " + tic + " " + name.iloc[0] + " ######")
print("###### " + gig.iloc[0] + " ######")
compEmbeddings = embeddings[embeddings['ticker']==tic][["topic"+str(x) for x in range(0,24)]].T.reset_index()
compEmbeddings.columns = ['topic','relevance']
compEmbeddings = compEmbeddings.merge(topicnames, on='topic')
compEmbeddings = compEmbeddings.sort_values(by='relevance', ascending=False)
compEmbeddings = compEmbeddings[['relevance','topic_words']]
for i,c in compEmbeddings.iterrows(): 
    print(str(round(c.relevance,3)) + " " + str(c.topic_words[0:5]))

def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):

        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2 # calculate the square of the difference of ith distr elements
    
        list_of_squares.append(s) # append that list

    sosq = sum(list_of_squares) # calculate sum of squares    

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
    
tic = "SIIX"

hDist = getCompetitive(tic,embeddings)
comparables = embeddings.merge(hDist, on='companyid')
comparables = comparables[['companyid','indGrp','topic','distance']]
comparables = comparables.sort_values(by='distance')
comparables = comparables.merge( longD[['companyid','compustat_name','ticker']], on='companyid' )
for i, c in comparables.head(n=25).iterrows():
    print(str(round(c['distance'],3)) + " " + str(c['compustat_name']) + " " + str(c['ticker']) + " " + str(c['topic']) + "    " + str(c['indGrp']) )

def printEmbeddings(ticker):
    name = embeddings[embeddings['ticker']==ticker]['compustat_name']
    gig = embeddings[embeddings['ticker']==ticker]['indGrp']
    print("###### " + ticker + " " + name.iloc[0] + " ######")
    print("###### " + gig.iloc[0] + " ######")
    compEmbeddings = embeddings[embeddings['ticker']==ticker][["topic"+str(x) for x in range(0,24)]].T.reset_index()
    compEmbeddings.columns = ['topic','relevance']
    compEmbeddings = compEmbeddings.merge(topicnames, on='topic')
    compEmbeddings = compEmbeddings.sort_values(by='relevance', ascending=False)
    compEmbeddings = compEmbeddings[['relevance','topic_words']]
    for i,c in compEmbeddings.iterrows(): 
        print(str(round(c.relevance,3)) + " " + str(c.topic_words[0:5]))

def wordFrequencyTicker(tic, col):
    sdFreq = longD[longD['ticker']==tic][col].str.split(expand=True).stack().value_counts()
    sdFreq = pd.DataFrame(sdFreq).reset_index()
    sdFreq.columns = ['word','n']
    return sdFreq

wfSIIX = wordFrequencyTicker("SIIX", 'cleanDesc')
wfDSGN = wordFrequencyTicker("DSGN", 'cleanDesc')
wfFL = wordFrequencyTicker("FL", 'cleanDesc')
wfCompare = wfSIIX.merge(wfDSGN, how='outer', on='word').fillna(0)
wfCompare.columns = ['word','n_DSGN','n_FL']
wfCompare = wfCompare.merge(wfFL, how='outer', on='word').fillna(0)
wfCompare.columns = ['word','n_SIIX','n_DSGN','n_FL']
    
printEmbeddings("SIIX")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(wfCompare.sort_values(by='n_SIIX', ascending=False).head(n=20))

printEmbeddings("DSGN")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(wfCompare.sort_values(by='n_DSGN', ascending=False).head(n=20))

printEmbeddings("FL")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(wfCompare.sort_values(by='n_FL', ascending=False).head(n=20))

pickle.dump( embeddings, open( "embeddings_unit3.p", "wb" ) )

print("Pickle dumped")
