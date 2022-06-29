from flask import Flask, render_template, request
import requests
from io import BytesIO
from PIL import Image
import pickle

import numpy as np
import pandas as pd
from datetime import datetime
import json
from glob import glob


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, OPTICS, DBSCAN, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture



def show_results_button(cluster_df, data, map_file):
    """_summary_

    Args:
        cluster_df (_type_): _description_
        data (_type_): _description_
        map_file (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if 'Author' not in data.columns:
        data['Author'] = data['AuthorOriginal']
    merged_cluster = cluster_df.merge(data[['uid', 'Author', 'AuthorOriginal', 'Description', 'annotated', 'Country', 'BeginDate']], left_on='uid', right_on='uid', how='left')
    merged_cluster['all'] = merged_cluster['Author'].astype(str) + ' '+ merged_cluster['AuthorOriginal'].astype(str) + ' ' + merged_cluster['Description'].astype(str) + ' ' + merged_cluster['Country'].astype(str) + ' ' + merged_cluster['BeginDate'].astype(str) + ' ' + merged_cluster['annotated'].astype(str)
    if request.method == "POST":
        if request.form["submit"] in ["text_search", "random_search", "next_search", "metadata_search"]:
            if request.form["submit"] == "text_search":
                cluster = [int(elt) for elt in request.form["item"].split(',')]
            elif request.form["submit"] == "metadata_search":
                cluster = list(merged_cluster[merged_cluster['all'].str.lower().str.contains(
                    request.form["item_meta"].lower())]['cluster'].unique())
            elif request.form["submit"] == "random_search":
                if 'cluster_size' in cluster_df.columns:
                    print('correct')
                    cluster = cluster_df[cluster_df['cluster_size'] > 1].groupby('cluster').first().sample(1).index.values    
                else:
                    cluster = cluster_df.groupby('cluster').first().sample(1).index.values
                    print(cluster)
            else:
                cluster = [int(elt) + 1 for elt in request.form["item"].split(',')]

            print(cluster_df[cluster_df['cluster'].isin(cluster)].shape)
        else:
            cluster = [int(elt) + 1 for elt in request.form["item"].split(',')]
    else:
        cluster = cluster_df.groupby('cluster').first().sample(1).index.values
      
    INFO = images_in_clusters(cluster_df[cluster_df['cluster'].isin(cluster)], data, map_file=map_file)
    return INFO, ','.join([str(cl) for cl in cluster])


def annotate_store(cluster_df, data, map_file, cluster_file, data_dir):
    """_summary_

    Args:
        cluster_df (_type_): _description_
        data (_type_): _description_
        map_file (_type_): _description_
        cluster_file (_type_): _description_
        data_dir (_type_): _description_
    """    
    if request.method == "POST":
        if request.form["submit"] in ["similar_images", "both_images", "wrong", "correct", "general_images", "both_general_images", ]:
            cluster = [int(elt) for elt in request.form["item"].split(',')]
            INFO = images_in_clusters(cluster_df[cluster_df['cluster'].isin(cluster)], data, map_file=map_file)
    
            imges_uids_sim = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    imges_uids_sim.append(request.form[form_key])
            cluster_num = int(request.form["form"])
            
            if request.form["submit"] == "similar_images":
                store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir)

            if request.form["submit"] == "both_images":
                store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir, negatives=True)

            if request.form["submit"] == "general_images":
                store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir, type_ann=['SIMILAR'], negatives=False)

            if request.form["submit"] == "both_general_images":
                store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir, type_ann=['SIMILAR', 'DIFFERENT'], negatives=True)

            if request.form["submit"] == "wrong":
                store_wrong_positive_cluster(INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir, wrong=True)
            
            if request.form["submit"] == "correct":
                store_wrong_positive_cluster(INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir, wrong=False)
            

def images_in_clusters(cluster_df, data, data_dir='../data/', map_file='map2pos_10-05-2022.pkl'):
    """_summary_

    Args:
        cluster_df (_type_): _description_
        data (_type_): _description_
        data_dir (str, optional): _description_. Defaults to '../data/'.
        map_file (str, optional): _description_. Defaults to 'map2pos_10-05-2022.pkl'.

    Returns:
        _type_: _description_
    """    
    data_agg = {}
    with open(data_dir + map_file, 'rb') as infile:
        map2pos = pickle.load(infile)

    if not 'annotated' in data.columns:
        data['annotated'] = ''

    if not 'Description (EN)' in data.columns:
        data['Description (EN)'] = ''

    if not 'Country' in data.columns:
        data['Country'] = ''
        
    if not 'City' in data.columns:
        data['City'] = ''
    
    cluster_df['pos'] = cluster_df['uid'].apply(lambda x: catch(x, map2pos))
    for cluster in cluster_df['cluster'].unique():
        if cluster not in [-1]:
            data_agg[int(cluster)] = []
            rows = cluster_df[cluster_df['cluster'] == cluster]
            for row in rows.iterrows():
                row_2 = data[data['uid'] == row[1]['uid']]
                if row_2.shape[0] > 0 and str(row_2["set"].values[0]) != 'nan':
                    if '2022' in str(row_2["annotated"].fillna('').astype(str).values[0]).split('_')[0].split(' ')[0]:
                        info_2 = '<b style="color:red">' + str(row_2["Author"].values[0]) + '<br> ' + str(row_2["Description"].values[0]#) + ' ' + str(row_2["Description (EN)"].values[0]
                            ) + '<br> Estimated time of production: ' + str(row_2["BeginDate"].values[0]
                            ) + '<br> Current location: ' + str(row_2["Country"].values[0]) + ' ' + str(row_2["City"].values[0]
                            ) + '<br> Annotated on: ' + str(row_2["annotated"].fillna('').astype(str).values[0]).split('_')[0].split(' ')[0] + ', in set: ' + str(row_2["set"].values[0]
                            ) + '</b>'    
                    else:
                        info_2 = '<b>' + str(row_2["Author"].values[0]) + '<br> ' + str(row_2["Description"].values[0]#) + ' ' + str(row_2["Description (EN)"].values[0]
                            ) + '<br> Estimated time of production: ' + str(row_2["BeginDate"].values[0]
                            ) + '<br> Current location: ' + str(row_2["Country"].values[0]) + ' ' + str(row_2["City"].values[0]
                            ) + '<br> Annotated on: ' + str(row_2["annotated"].fillna('').astype(str).values[0]).split('_')[0].split(' ')[0] + ', in set: ' + str(row_2["set"].values[0]
                            ) + '</b>' 
                elif row_2.shape[0] > 0:
                    info_2 = str(row_2["Author"].values[0]) + '<br> ' + str(row_2["Description"].values[0]#) + ' ' + str(row_2["Description (EN)"].values[0]
                            ) + '<br> Estimated time of production: ' + str(row_2["BeginDate"].values[0]
                            ) + '<br> Current location: ' + str(row_2["Country"].values[0]) + ' ' + str(row_2["City"].values[0]
                            ) + '<br> Annotated on: ' + str(row_2["annotated"].fillna('').astype(str).values[0]).split('_')[0].split(' ')[0] + ', in set: ' + str(row_2["set"].values[0]
                            )
                else:
                    info_2 = ''
                uid = row[1]['uid']
                pos = row[1]['pos']    
                
                if row_2.shape[0] > 0 and 'ImageURL' in data.columns:
                    url = row_2['ImageURL'].values[0]
                    if 'html' in url: 
                            
                        image = url.split('html')[0]+'art'+url.split('html')[1] +'jpg'
                    else:
                        image = url
                elif 'WGA' in row[1]['path']:
                    drawer = '/'.join(row[1]['path'].split('/')[6:]).split('.')[0] # http://www.wga.hu/html/a/aachen/allegory.html
                    image = f'http://www.wga.hu/art/{drawer}.jpg'
                else:
                    try:
                        drawer = row[1]['path'].split('/')[-1].split('_')[0]
                        img = row[1]['path'].split('/')[-1].split('_')[1].split('.')[0]
                        image = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
                    except:
                        image = ''
                      
                data_agg[int(cluster)].append([info_2, image, uid, float(pos[0]), float(pos[1])])
    return data_agg
    


def make_clusters_embeddings(data_dir='../data/', data_file='data_wga_cini_45000.csv', 
                             embed_file='resnext-101_epoch_410-05-2022_10%3A11%3A05.npy', dist=0.13,
                             min_n=2, type_clustering='dbscan', dist2=0.12):
    
    """_summary_

    Returns:
        _type_: _description_
    """    
    data = pd.read_csv(data_dir + data_file)
    data = data.groupby(['Description', 'AuthorOriginal']).first().reset_index()
    embeds = np.load(data_dir + embed_file, allow_pickle=True) 
    print(embeds.shape)
    embeds = embeds[np.in1d(embeds[:, 0], list(data['uid'])),:]
    print(embeds.shape)
    uids = list(data['uid'])
    
    uid2path = {}
    for i, row in data.iterrows():
        uid2path[row['uid']] = row['path']
    
    if type_clustering=='dbscan':
        if embeds.shape[0] > 80000:
            uid2remove = []
            for i in [(0, 10000),(10000, 20000), (20000, 30000), (30000, 40000),
                       (40000, 50000), (50000, 60000), (60000, 70000),(70000, embeds.shape[0])]:
                print(i)
                if i[1] == 80000:
                    emb = np.concatenate((np.vstack(embeds[i[0]:i[1],1]), np.vstack(embeds[-5000:,1])), axis=0)
                    labels = np.concatenate((embeds[i[0]:i[1],0], embeds[-5000:,0]), axis=0)
                
                else:
                    emb = np.vstack(embeds[i[0]:i[1],1])
                    labels = embeds[i[0]:i[1],0]
                print(emb.shape)
                db = DBSCAN(eps=dist2, min_samples=1, metric='cosine').fit(emb) #0.51 best so far
                classes = db.labels_
                clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
                print(clusters['cluster'].value_counts())
                uid2remove.append(list(clusters[clusters['cluster'] == -1]['uid']))
                print(len(uid2remove[-1]))

            uid2remove = [i for in_ in uid2remove for i in in_ ]
            new_embs = embeds[~np.in1d(embeds[:, 0], uid2remove),1]
            print(new_embs.shape)
            db = DBSCAN(eps=dist, min_samples=min_n, metric='cosine').fit(np.vstack(new_embs)) #0.51 best so far
            classes = db.labels_
            labels = embeds[~np.in1d(embeds[:, 0], uid2remove), 0]

            

        else:   
            db = DBSCAN(eps=dist, min_samples=min_n, metric='cosine').fit(np.vstack(embeds[:,1])) #0.51 best so far
            classes = db.labels_
            labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
    elif type_clustering=='gaussian_mixture':
        embeddings_new = PCA(n_components=20).fit_transform(
            np.vstack(embeds[:, 1])
        )
        print('dim reduction done')
        gm = BayesianGaussianMixture(n_components=dist).fit(embeddings_new)
        classes = gm.predict(embeddings_new)
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
    elif type_clustering=='kmeans_dim':
        embeddings_new = PCA(n_components=20).fit_transform(
            np.vstack(embeds[:, 1])
        )
        print('dim reduction done')
        km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[:,1]))
        classes = km.labels_
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
    elif type_clustering == 'optics':
        # embeddings_new = PCA(n_components=500).fit_transform(
        #     np.vstack(embeds[:, 1])
        # )
        # print('dim reduction done')
        
        # db = OPTICS(max_eps=dist, min_samples=min_n, metric='cosine').fit(np.vstack(embeddings_new)) #0.51 best so far
        db = OPTICS(max_eps=dist, min_samples=min_n, metric='cosine').fit(embeds[:, 1]) #0.51 best so far
        classes = db.labels_
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
        
    elif type_clustering == 'mix':
        print('remove outliers with dbscan and cluster with kmeans')
        print(dist2)
        db = DBSCAN(eps=dist2, min_samples=min_n, metric='cosine').fit(np.vstack(embeds[:,1])) #0.51 best so far
        classes = db.labels_
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
        uid2remove = list(clusters[clusters['cluster'] == -1]['uid'])
        print(len(uid2remove))
        
        # embeddings_new = TSNE(n_components=2).fit_transform(
        #     np.vstack(embeds[~np.in1d(embeds[:, 0], uid2remove),1])
        # )
        # print('dim reduction done')
        # km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeddings_new[:,1]))
        
        km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[~np.in1d(embeds[:, 0], uid2remove),1]))
        
        #km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[~np.in1d(embeds[:, 0], uid2remove),1]))
        classes = km.labels_
        labels = embeds[~np.in1d(embeds[:, 0], uid2remove), 0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
    elif type_clustering == 'dbscan_kmeans':
        db = DBSCAN(eps=dist2, min_samples=min_n, metric='cosine').fit(np.vstack(embeds[:,1])) #0.51 best so far
        classes = db.labels_
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
        print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
        unique = clusters['cluster'].nunique()
        max_cluster = clusters[clusters['cluster'] != -1].groupby(['cluster']).size().idxmax()
        print(max_cluster)
        uid2keep = list(clusters[clusters['cluster'] == max_cluster]['uid'])
        print(len(uid2keep))
        km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[np.in1d(embeds[:, 0], uid2keep),1]))
        classes_new = km.labels_
        print(classes_new)
        #labels_new = embeds[np.in1d(embeds[:, 0], uid2keep), 0]
        clusters.loc[clusters['cluster'] == max_cluster, 'cluster'] = classes_new + unique
        
    elif type_clustering == 'spectral_clustering':
        clustering = SpectralClustering(n_clusters=dist,
                                        assign_labels='discretize',
                                        random_state=0,
                                        ).fit(np.vstack(embeds[:,1]))
        classes = clustering.labels_
        labels = embeds[:,0]
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    
    
    else:
        km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[:,1]))
        classes = km.labels_
        labels = embeds[:,0]
        
        clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    print(clusters.shape)
    print('stats')
    print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    
    clusters = clusters[clusters['uid'].isin(uids)].reset_index()
    print(clusters.shape)
    clusters['path'] = clusters['uid'].apply(lambda x: uid2path[x])
    clu2size = {i: cl for i,cl in zip(clusters.groupby('cluster').size().index, clusters.groupby('cluster').size().values)}
    clusters['cluster_size'] = clusters['cluster'].apply(lambda x: clu2size[x])

    return clusters



def get_2d_pos(data_dir='../data/', embed_file='resnext-101_epoch_410-05-2022_10%3A11%3A05.npy'):
    """_summary_

    Args:
        data_dir (str, optional): _description_. Defaults to '../data/'.
        embed_file (str, optional): _description_. Defaults to 'resnext-101_epoch_410-05-2022_10%3A11%3A05.npy'.

    Returns:
        _type_: _description_
    """    
    embeds = np.load(data_dir + embed_file, allow_pickle=True) 
    embeddings_new = TSNE(
            n_components=2
        ).fit_transform(np.vstack(embeds[:, 1]))
    map2pos = {}
    for i, uid in enumerate(embeds[:,0]):
        map2pos[uid] = embeddings_new[i]
    return map2pos



def store_wrong_positive_cluster(info_cluster, cluster_num, cluster_file, data_dir='/scratch/students/schaerf/annotation/', wrong=True):
    """_summary_

    Args:
        info_cluster (_type_): _description_
        cluster_num (_type_): _description_
        cluster_file (_type_): _description_
        data_dir (str, optional): _description_. Defaults to '/scratch/students/schaerf/annotation/'.
        wrong (bool, optional): _description_. Defaults to True.
    """    
    morpho = pd.read_csv(data_dir + 'morphograph_clusters_new.csv')

    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")

    if wrong:
        tpl = ('NEGATIVE', 'WRONG' )
    else:
        tpl = ('POSITIVE', 'CORRECT')
    
    
    to_add = []
    for info in info_cluster:
        if info[0][-3:] in ['ain', 'est', 'val']:
            for info_2 in info_cluster:
                if not info_2[0][-3:] in ['ain', 'est', 'val']:
                        to_add.append([info[2][:16]+info_2[2][16:], info[2], info_2[2], tpl[0], now, cluster_file, cluster_num])
        else:
            for info_2 in info_cluster:
                if info[2] != info_2[2]:
                        to_add.append([info[2][:16]+info_2[2][16:], info[2], info_2[2], tpl[1], now, cluster_file, cluster_num])

    print(to_add)
    new_morphs = pd.DataFrame(to_add, columns=['uid_connection', 'img1', 'img2', 'type', 'date', 'cluster_file', 'cluster'])
    update = pd.concat([morpho, new_morphs], axis=0)
    print(update[['uid_connection', 'type', 'cluster']].tail())
    print(morpho.shape, update.shape)
    update.to_csv(data_dir + 'morphograph_clusters_new.csv', index=False)


def store_morph_cluster(imges_uids_sim, info_cluster, cluster_num, cluster_file, data_dir='/scratch/students/schaerf/annotation/', type_ann=['POSITIVE', 'NEGATIVE'], negatives=False):
    """_summary_

    Args:
        imges_uids_sim (_type_): _description_
        info_cluster (_type_): _description_
        cluster_num (_type_): _description_
        cluster_file (_type_): _description_
        data_dir (str, optional): _description_. Defaults to '/scratch/students/schaerf/annotation/'.
        type_ann (list, optional): _description_. Defaults to ['POSITIVE', 'NEGATIVE'].
        negatives (bool, optional): _description_. Defaults to False.
    """    
    morpho = pd.read_csv(data_dir + 'morphograph_clusters_new.csv')
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    
    to_add = []
    
    for uid in imges_uids_sim:
        for info in info_cluster:
            if uid == info[2] and not info[0][-3:] in ['ain', 'est', 'val']:
                for uid2 in imges_uids_sim:
                    if uid2 != uid:
                        to_add.append([uid[:16]+uid2[16:], uid, uid2, type_ann[0], now, cluster_file, cluster_num])

    if negatives:
        
        different_images = [info[2] for info in info_cluster if info[2] not in imges_uids_sim] 
        for uid in imges_uids_sim:
            for uid2 in different_images:
                to_add.append([uid[:16]+uid2[16:], uid, uid2, type_ann[1], now, cluster_file, cluster_num])

    new_morphs = pd.DataFrame(to_add, columns=['uid_connection', 'img1', 'img2', 'type', 'date', 'cluster_file', 'cluster'])
    update = pd.concat([morpho, new_morphs], axis=0)
    print(update[['uid_connection', 'type', 'cluster']].tail())
    print(morpho.shape, update.shape)
    update.to_csv(data_dir + 'morphograph_clusters_new.csv', index=False)



def make_clusters_rerank(data_dir='../data/', uid2path_file = 'uid2path.pkl', final_file='list_iconography.pkl', embed_file='similarities_madonnas_2600.npy'):
    """_summary_

    Args:
        data_dir (str, optional): _description_. Defaults to '../data/'.
        uid2path_file (str, optional): _description_. Defaults to 'uid2path.pkl'.
        final_file (str, optional): _description_. Defaults to 'list_iconography.pkl'.
        embed_file (str, optional): _description_. Defaults to 'similarities_madonnas_2600.npy'.

    Returns:
        _type_: _description_
    """    
    with open(data_dir + uid2path_file, 'rb') as outfile:
        uid2path = pickle.load(outfile)
    with open(data_dir + final_file, 'rb') as infile:
        final = pickle.load(infile)
    
    sim_mat = np.load(data_dir + embed_file, allow_pickle=True) #embedding_no_pool/)
    diff_mat = np.round(1 - sim_mat, 3)
    db = DBSCAN(eps=0.03, min_samples=2, metric='precomputed').fit(diff_mat)
    labels = final[:sim_mat.shape[0]]
    classes = db.labels_

    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    #print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    clusters['path'] = clusters['uid'].apply(lambda x: catch(x, uid2path))

    return clusters


def make_links(data_hierarchical):
    """_summary_

    Args:
        data_hierarchical (_type_): _description_

    Returns:
        _type_: _description_
    """    
    cluster_lists = data_hierarchical.groupby('cluster_desc')['cluster'].apply(lambda x: list(x))
    pairs_to_match = []
    for list_ in cluster_lists:
        for i in range(len(list_)):
            for j in range(len(list_) - i):
                if list_[j] != list_[i]:
                    if list_[j] != -1 and list_[i] != -1:
                        pairs_to_match.append(list(set([str(list_[i]),str(list_[j])])))
    return list(set(['-'.join(pair) for pair in pairs_to_match]))


def catch(x, uid2path):
    """_summary_

    Args:
        x (_type_): _description_
        uid2path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    try:
        return uid2path[x]
    except:
        return [0,0]

def convert_to_json(data_agg):
    """_summary_

    Args:
        data_agg (_type_): _description_

    Returns:
        _type_: _description_
    """        
    new = ''
    for cluster in data_agg.keys():
        new  += '!!' + str(cluster) + '%%' + '%%'.join(['$$'.join([str(c) for c in cli]) for cli in data_agg[cluster]])
    return new
