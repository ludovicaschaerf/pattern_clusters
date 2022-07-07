import pandas as pd
import numpy as np
import pickle
import sys
from tqdm import tqdm
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
    
from .utils_clusters import *
from .metrics_clusters import *
from .utils import *

def cluster_text(text, range_try=(100,102), hyperparam=False, num_clusters=100):
    """Kmean++ clustering based on Manhattan distance.

    Args:
        text (_type_): _description_
        range_try (tuple, optional): _description_. Defaults to (100,102).
        hyperparam (bool, optional): _description_. Defaults to False.
        num_clusters (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """    
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(text)


    #import matplotlib.pyplot as plt
    
    if hyperparam:
        Sum_of_squared_distances = []
        K = range(range_try[0],range_try[1])
        for k in K:
            km = KMeans(n_clusters=k, max_iter=200, n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        print('How many clusters do you want to use?')
        true_k = int(input())
    else:
        true_k = int(num_clusters)
    
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)

    labels=model.labels_
    clusters=pd.DataFrame(list(zip(text,labels)),columns=['title','cluster'])
    print(clusters.sort_values(by=['cluster']))
        
    return clusters


def add_interest_scores(data_dir='../data/', translate=False, new=True, precomputed=True):
    """Adds multiple scores for each cluster based on its Iconographic, Author, Attrbitues, Place, Time variance.
    The attribute and iconographic variances are based on the number of text-clusters in each morphograph group
    of the author attributes text (ie. (attr.)) and the cleaned descriptor. The author is based on the number of 
    different lastnames and the place and time on the maximal variance in date of begin of the work and latitute 
    and longitude for the place.

    Args:
        data_dir (str, optional): _description_. Defaults to '../data/'.
        translate (bool, optional): _description_. Defaults to False.
        new (bool, optional): _description_. Defaults to False.
        precomputed (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """    
    positives = update_morph(data_dir, '', new=new) 
    
    if precomputed:
        pos = pd.read_csv(data_dir + 'interest_scores.csv')
        positives = positives.merge(pos, left_on=['uid', 'cluster'], right_on=['uid', 'cluster'], how='left')
        return positives
    # scores for iconography
    if translate:
        positives.loc[:, 'Description (EN)'] = [catch_transl(lambda : GoogleTranslator(source='auto', target='en').translate(word)) 
                                      for word in tqdm(list(positives.loc[:, 'Description'].astype(str)))]

    else:
        with open(data_dir + 'uid2desc.pkl', 'rb') as infile:
            uid2endesc = pickle.load(infile)

        positives.loc[:, 'Description (EN)'] = [catch_transl_2(uid, word, uid2endesc) 
                                      for uid, word in tqdm(zip(list(positives.loc[:, 'uid'].astype(str)), list(positives.loc[:, 'Description'].astype(str))))]

        uid2endesc = {uid:desc for uid,desc in zip(positives['uid'], positives['Description (EN)'])}

        with open(data_dir + 'uid2desc.pkl', 'wb') as outfile:
            pickle.dump(uid2endesc, outfile)

    positives['Description (EN - ref)'] = positives['Description (EN)'].astype(str).str.split(r'[^S-]\.').apply(lambda x: x[0]).str.split(',').apply(lambda x: x[0]).apply(lambda x: x.replace('0123456789', ''))
    print(positives['Description (EN - ref)'].value_counts())
    clusters = cluster_text(positives['Description (EN - ref)'].values, num_clusters=50)
    clusters['cluster_iconography'] = clusters['cluster']

    positives = positives.merge(clusters[['cluster_iconography']], left_index=True, right_index=True)
    scores_iconography = {cluster: np.around(content['cluster_iconography'].nunique() - content.shape[0] * 0.1,2) for cluster, content in positives.groupby('cluster') if content.shape[0] > 1}
    positives['scores_iconography'] = positives['cluster'].apply(lambda x: scores_iconography[x] if x in scores_iconography.keys() else 0)
    
    ## scores for authors
    positives['AuthorClean'] = positives['Author'].str.split().apply(lambda x: x[0])
    scores_authors = {cluster: np.around(content[content['AuthorClean'].notnull()]['AuthorClean'].nunique() - content.shape[0] * 0.01,2) for cluster, content in positives.groupby('cluster') if content[content['AuthorClean'].notnull()].shape[0] > 1}
    positives['scores_authors'] = positives['cluster'].apply(lambda x: scores_authors[x] if x in scores_authors.keys() else 0)
    
    positives['AuthorAttr'] = positives['AuthorOriginal'].str.split('(').apply(lambda x: x[1] if len(x)>1 else 'Original').str.split(')').apply(lambda x: x[0]).apply(lambda x: x.replace('-)', '')).apply(lambda x: x.strip(') '))
    clusters = cluster_text(positives['AuthorAttr'].values, num_clusters=5)
    clusters['cluster_attribution'] = clusters['cluster']

    ### scores for different attributions
    positives = positives.merge(clusters[['cluster_attribution']], left_index=True, right_index=True)
    scores_attributions = {cluster: np.around(content['cluster_attribution'].nunique() - content.shape[0] * 0.05,2) for cluster, content in positives.groupby('cluster') if content.shape[0] > 1}
    positives['scores_attributions'] = positives['cluster'].apply(lambda x: scores_attributions[x] if x in scores_attributions.keys() else 0)
    
    ## scores for time
    extras = pd.read_csv(data_dir + 'morphograph/Cini_AllVariationsMerged_20210421.csv', sep=';')
    positives = positives.merge(extras[['Author', 'AuthorULAN', 'AuthorULANLabel', 'AuthorNationality',  'BiographyLabel', 
                                          'AuthorDeath', 'AuthorBirthLong', 'AuthorBirthLat', 'AuthorDeathLong', 'AuthorDeathLat',
                                           'CountModifiers',]], left_on='Author', right_on='Author', how='left')
    scores_times = {cluster: np.around(content[content['BeginDate'].notnull()]['BeginDate'].max() - content[content['BeginDate'].notnull()]['BeginDate'].min() + (content.shape[0] * 0.05), 2) for cluster, content in positives.groupby('cluster') if content[content['BeginDate'].notnull()].shape[0] > 1}
    positives['scores_times'] = positives['cluster'].apply(lambda x: scores_times[x]  if x in scores_times.keys() else 0)
    
    ## scores for place
    scores_places = {cluster: np.around(content[content['AuthorDeathLat'].notnull()]['AuthorDeathLat'].max() - content[content['AuthorDeathLat'].notnull()]['AuthorDeathLat'].min() + content[content['AuthorDeathLong'].notnull()]['AuthorDeathLong'].max() - content[content['AuthorDeathLong'].notnull()]['AuthorDeathLong'].min() + (content.shape[0] * 0.05), 2) for cluster, content in positives.groupby('cluster') if content[content['AuthorDeathLat'].notnull()].shape[0] > 1}
    positives['scores_places'] = positives['cluster'].apply(lambda x: scores_places[x]  if x in scores_places.keys() else 0)
    
    clu2count = {cl:group.shape[0] for cl, group in positives.groupby('cluster')}
    positives['scores_count'] = positives['cluster'].apply(lambda x: clu2count[x])
    positives[['uid', 'cluster', 'Description (EN)',
       'Description (EN - ref)', 'cluster_iconography', 'AuthorClean',
       'AuthorAttr', 'cluster_attribution', 'scores_iconography',
       'scores_authors', 'scores_attributions', 'AuthorDeathLong',
       'AuthorDeathLat', 'CountModifiers', 'scores_times', 'scores_places',
       'scores_count']].to_csv(data_dir + 'interest_scores.csv')
    print(positives.columns)

    return positives


def catch_transl(func, zero=False, handle=lambda e: e, *args, **kwargs, ):
    """Prevents list comprehensions from going into an error when an exception occurs

    Returns:
        _type_: _description_
    """    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        if zero:
            return 0
        else:
            return e

def catch_transl_2(uid, word, uid2endesc):
    """Prevents list comprehensions from going into an error when an exception occurs

    Returns:
        _type_: _description_
    """    
    try:
        if 'Max retries' in uid2endesc[uid]:
            return catch_transl(lambda : GoogleTranslator(source='auto', target='en').translate(word))
        else:
            return uid2endesc[uid]
    except Exception as e:
        return catch_transl(lambda : GoogleTranslator(source='auto', target='en').translate(word))

