import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree

import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch import nn
import torchvision.models as models
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
import requests
from io import BytesIO

#from metrics import recall_at_k, mean_average_precision
from PIL import Image

#from spatial_reranking import rerank_spatial

from textwrap import wrap


#########################################################
##### Embeddings
#########################################################

def get_embedding(img, model, type="fine_tune", device="cpu"):
    if type == "fine_tune":
        embedding = model.predict(img.squeeze(1).to(device))[0].cpu().detach().numpy()
        #print(embedding.shape)
    else:
        embedding = model(img.squeeze(1).to(device))[0].cpu().detach().numpy()
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
    return embedding

def get_lower_dimension(embeddings, dimensions=100, method="umap"):
    if method == "umap":
        embeddings_new = umap.UMAP(n_components=dimensions, metric="cosine").fit(
            np.vstack(embeddings[:, 1])
        )
        embeddings_new = embeddings_new.embedding_
    elif method == "pca":
        embeddings_new = PCA(n_components=dimensions).fit_transform(
            np.vstack(embeddings[:, 1])
        )
    elif method == "svd":
        embeddings_new = TruncatedSVD(n_components=dimensions).fit_transform(
            np.vstack(embeddings[:, 1])
        )
    elif method == "tsne":
        embeddings_new = TSNE(
            n_components=dimensions
        ).fit_transform(np.vstack(embeddings[:, 1]))
    else:
        print("unknow method")
        embeddings_new = embeddings

    return embeddings_new


def make_tree_orig(embeds, reverse_map=False):
    if reverse_map:
        kdt = BallTree(np.vstack(embeds[:, 1]), metric="euclidean")
        reverse_map = {k: embeds[k, 0] for k in range(len(embeds))}
        return kdt, reverse_map
    else:
        kdt = BallTree(np.vstack(embeds[:, 1]), metric="euclidean")
        return kdt


def find_most_similar_orig(uid, tree, embeds, uids, n=401, similarities=False):
    img = np.vstack(embeds[embeds[:, 0] == uid][:, 1]).reshape(1, -1)
    cv_all = tree.query(img, k=n)#[1][0]
    cv = cv_all[1][0]
    if similarities:
        similarities = cv_all[0][0]
        return [uids[c] for c in cv if uids[c] != uid], similarities[1:]
    else:
        return [uids[c] for c in cv if uids[c] != uid]


def find_most_similar_no_theo(uid, tree, embeds, uids, list_theo, n=401):
    img = np.vstack(embeds[embeds[:, 0] == uid][:, 1]).reshape(1, -1)
    cv = tree.query(img, k=n)[1][0]
    return [uids[c] for c in cv if uids[c] not in list_theo]  # not in uids_match

#########################################################
##### Retrieval
#########################################################

def find_pos_matches(uids_sim, uids_match, how="all", n=400):
    matched = list(filter(lambda i: uids_sim[i] in uids_match, range(len(uids_sim))))
    while len(matched) < len(uids_match):
        matched.append(n)
    if how == "all":
        if len(matched) > 0:
            return matched
        else:
            return [n]
    elif how == "first":
        if len(matched) > 0:
            return matched[0]
        else:
            return n
    elif how == "median":
        if len(matched) > 0:
            return np.median(np.array(matched))
        else:
            return n


def make_rank(uids_sim, uids_match):
    return [1 if uid in uids_match else 0 for uid in uids_sim]


def catch(x, uid2path):
    try:
        return uid2path[x]
    except Exception as e:
        #print(e)
        return np.nan

    
#########################################################
##### Preprocess data
#########################################################


def get_train_test_split(metadata, morphograph):
    if 'cluster_file' not in morphograph.columns:
        morphograph['cluster_file'] = 'Original'


    positives = morphograph[morphograph["type"] == "POSITIVE"]
    positives.columns = ["uid_connection", "img1", "img2", "type", "annotated", "cluster_file"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        positives,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    # merging the two files
    positives = pd.concat(
        [
            metadata.merge(positives, left_on="uid", right_on="img1", how="inner"),
            metadata.merge(positives, left_on="uid", right_on="img2", how="inner"),
        ],
        axis=0,
    ).reset_index()

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positives["cluster"] = positives["uid"].apply(lambda x: mapper[x])
    positives["set"] = [
        "test" if cl % 3 == 0 else "train" for cl in positives["cluster"]
    ]

    positives["set"] = [
        "val" if cl % 6 == 0 else set_ for cl, set_ in zip(positives["cluster"], positives["set"])
    ]

    return positives


def remove_duplicates(metadata, morphograph):

    positives = morphograph[morphograph["type"] == "DUPLICATE"]
    positives.columns = ["uid_connection", "img1", "img2", "type", "annotated"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        positives,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    # merging the two files
    positives = pd.concat(
        [
            metadata.merge(positives, left_on="uid", right_on="img1", how="outer"),
            metadata.merge(positives, left_on="uid", right_on="img2", how="outer"),
        ],
        axis=0,
    ).reset_index()

    # adding set specification to df
    mapper = {it: it for it in positives["uid"].unique()}

    for number, nodes in enumerate(components):
        for it in nodes:
            mapper[it] = number

    positives["cluster"] = positives["uid"].apply(lambda x: mapper[x])
    positives = positives.groupby("cluster").first().reset_index()
    return positives


#########################################################
##### Image processing
#########################################################


def show_images(img_names):
    f, axarr = plt.subplots(1, 3, figsize=(20, 10))
    axarr = axarr.flatten()
    for i, name in enumerate(img_names):
        img = Image.open(name)
        axarr[i].imshow(img)

    plt.show()


def preprocess_image(img_name, resolution=480):
    img = Image.open(img_name)
    img = img.convert('RGB')
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resolution+100, resolution+100)),
            transforms.RandomResizedCrop((resolution, resolution), ),
            transforms.ColorJitter(
                   brightness=0.3,
                   contrast=0.3,
                   saturation=0.3
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Grayscale(num_output_channels=3),
        ]
    )
    return tfms(img).unsqueeze(0)

def preprocess_image_test(img_name, resolution=480):
    img = Image.open(img_name)
    img = img.convert('RGB')
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomResizedCrop((resolution, resolution), ),
            transforms.Resize((resolution, resolution)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfms(img).unsqueeze(0)

#########################################################
##### Evalute and create new training
#########################################################

def get_scores(embeddings, train_test, data, list_downloaded=False, reverse_map=False, set_split=['test', 'val']):
    if reverse_map:
        tree, reverse_map = make_tree_orig(embeddings, True)
    
    else:
        tree = make_tree_orig(embeddings)
        reverse_map = list(data['uid'].unique())
    Cs = []
    Bs = []
    pos = []
    ranks = []

    if not list_downloaded:
        list_downloaded = list(train_test["img1"]) + list(train_test["img2"])

    for i in tqdm(range(train_test.shape[0])):
        if (train_test["img1"][i] in list_downloaded) and (train_test["img2"][i] in list_downloaded
            ) and (train_test["set"][i] in set_split):
            list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
            Bs.append(list_theo)
            
            list_sim = find_most_similar_orig(
                train_test["uid"][i], tree, embeddings, reverse_map, n=min(data.shape[0], 4000)
            )
            Cs.append(list_sim[:400])
            matches = find_pos_matches(list_sim[:400], list_theo, how="all")
            pos.append(matches)
            rank = make_rank(list_sim, list_theo)
            ranks.append(rank)
            
    posses = [po for p in pos for po in p]
    posses_min = [p[0] for p in pos]
    posses_med = [np.median(np.array(p)) for p in pos]

    mean_position = np.mean(np.array(posses))
    mean_min_position = np.mean(np.array(posses_min))
    mean_median_position = np.mean(np.array(posses_med))
            
    print('all positions', mean_position)
    print('min positions', mean_min_position)
    print('median positions', mean_median_position)

    map = mean_average_precision(ranks)
    print('mean average precision', map)

    recall_400 = np.mean([recall_at_k(ranks[i], 400) for i in range(len(ranks))])
    recall_200 = np.mean([recall_at_k(ranks[i], 200) for i in range(len(ranks))])
    recall_100 = np.mean([recall_at_k(ranks[i], 100) for i in range(len(ranks))])
    recall_50 = np.mean([recall_at_k(ranks[i], 50) for i in range(len(ranks))])
    recall_20 = np.mean([recall_at_k(ranks[i], 20) for i in range(len(ranks))])
    print('recall @ 400', recall_400)
    print('recall @ 200', recall_200)
    print('recall @ 100', recall_100)
    print('recall @ 50', recall_50)
    print('recall @ 20', recall_20)

    return mean_position, mean_min_position, mean_median_position, map, recall_400, recall_200, recall_100, recall_50, recall_20

def get_positions(embeddings, train_test, data, list_downloaded=False, reverse_map=False, set_split=['test', 'val']):
    if reverse_map:
        tree, reverse_map = make_tree_orig(embeddings, True)
    
    else:
        tree = make_tree_orig(embeddings)
        reverse_map = list(data['uid'].unique())
    Cs = []
    Bs = []
    pos = []
    ranks = []

    if not list_downloaded:
        list_downloaded = list(train_test["img1"]) + list(train_test["img2"])

    for i in tqdm(range(train_test.shape[0])):
        if (train_test["img1"][i] in list_downloaded) and (train_test["img2"][i] in list_downloaded
            ) and (train_test["set"][i] in set_split):
            list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
            Bs.append(list_theo)
            
            list_sim = find_most_similar_orig(
                train_test["uid"][i], tree, embeddings, reverse_map, n=min(data.shape[0], 400)
            )
            Cs.append(list_sim[:400])
            matches = find_pos_matches(list_sim[:400], list_theo, how="all")
            pos.append(matches)
            rank = make_rank(list_sim, list_theo)
            ranks.append(rank)
            
    posses = [np.mean(np.array(p)) for p in pos]
    posses_min = [p[0] for p in pos]
    posses_med = [np.median(np.array(p)) for p in pos]

    return posses, posses_min, posses_med

def make_training_set_orig(embeddings, train_test, data, data_dir, uid2path, epoch=10, n=10):
    tree = make_tree_orig(embeddings)
    Cs = []
    for i in tqdm(range(train_test.shape[0])):
        list_theo = (
            list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
            + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
            + [train_test["uid"][i]]
        )
        list_sim = find_most_similar_no_theo(
            train_test["uid"][i], tree, embeddings, list(data["uid"].unique()), list_theo, n=n+1
        )
        Cs.append(list_sim)
    
    train_test['C'] = Cs

    final = train_test[['img1', 'img2', 'C', 'set']].explode('C')
    final.columns = ['A', 'B', 'C', 'set']
    final['A_path'] = final['A'].apply(lambda x: catch(x, uid2path))
    final['B_path'] = final['B'].apply(lambda x: catch(x, uid2path))
    final['C_path'] = final['C'].apply(lambda x: catch(x, uid2path))
    
    final = final[final['C_path'].notnull() & final['A_path'].notnull() & final['B_path'].notnull()]#.sample(frac=0.5)
    print(final.shape)
    print(final.tail())

    print(epoch)
    final[final['set'] == 'train'].reset_index().to_csv(data_dir + 'dataset/abc_train_' + str(epoch) + '.csv')
    final[final['set'] == 'test'].reset_index().to_csv(data_dir + 'dataset/abc_test_' + str(epoch) + '.csv')
    final[final['set'] == 'val'].sample(frac=0.1).reset_index().to_csv(data_dir + 'dataset/abc_val_' + str(epoch) + '.csv')
    
    return final


#########################################################
##### Show results
#########################################################

def get_drawer(row_2, path, info=False):
    if type(row_2) != str and 'ImageURL' in row_2.columns:
        url = row_2['ImageURL'].values[0]
        if 'html' in url: 
                            
            image = url.split('html')[0]+'art'+url.split('html')[1] +'jpg'
        else:
            image = url
    elif 'WGA' in path:
        drawer = '/'.join(path.split('/')[4:]).split('.')[0] # http://www.wga.hu/html/a/aachen/allegory.html
        image = f'http://www.wga.hu/art/{drawer}.jpg'
    elif type(row_2) == str and row_2 == path:
        url = path
        if 'html' in url: 
                            
            image = url.split('html')[0]+'art'+url.split('html')[1] +'jpg'
        else:
            image = url
    else:
        try:
            drawer = path.split('/')[-1].split('_')[0]
            img = path.split('/')[-1].split('_')[1].split('.')[0]
            image = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
            if info:
                return drawer+'_'+img+'.jpg'
        
        except:
            image = ''
    
    return image
                
def show_similars(row, embeddings, train_test, tree, reverse_map, uid2path, data):
    
    if row["set"].values[0] in ['train', 'val', 'test']:
        list_theo = (
            list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
            + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
            + [row["uid"].values[0]]
        )
    else:
        raise Exception('File not in grountruth')
        
    sim, similarities = find_most_similar_orig(
        row["uid"].values[0], tree, embeddings, reverse_map, n=50+1, similarities=True
    )

    print('positions of matches', find_pos_matches(sim, list_theo, n=50))

    f, axarr = plt.subplots(3,4, figsize=(30,20))
    plt.suptitle('Image retrieval with embeddings of ' + row["uid"].values[0] + 
                '\nRank: ' + ', '.join([str(x) for x in find_pos_matches(sim, list_theo, n=50)]))
    axarr = axarr.flatten()
    
    image_a = requests.get(get_drawer(row_2, row[1]['path']))
        
    axarr[0].set_facecolor('black')
    axarr[0].imshow(Image.open(BytesIO(image_a.content))) #replica_dir + 
    axarr[0].set_title(row["AuthorOriginal"].values[0] + '\n ' + row["Description"].values[0] + '\n REFERENCE')
    for i in range(len(sim[:11])):
        if sim[i] in list_theo:
            correct = 'CORRECT'
        else:
            correct = 'WRONG'

        row_2 = data[data['uid'] == sim[i]]
        info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0])
    
        axarr[i+1].set_title(info_2 + '\n' + 'Similarity: ' + str(np.round((1 - similarities[i]), 2)) + ', Grountruth: ' + correct)
        image = requests.get(get_drawer(row_2, catch(sim[i], uid2path)))
        
        axarr[i+1].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.savefig('/scratch/students/schaerf/figures/' + row["uid"].values[0] + '.jpg')
    plt.show()
    
    sim_rerank, similarities_rerank = rerank_spatial(row["uid"].values[0], sim, uid2path, similarities=True)
    print('positions of matches after reranking', find_pos_matches(sim_rerank, list_theo, n=50))

    f, axarr = plt.subplots(3,4, figsize=(30,20))
    plt.suptitle('Image retrieval with reranking of ' + row["uid"].values[0] + 
                '\nRank: ' + ', '.join([str(x) for x in find_pos_matches(sim_rerank, list_theo, n=50)]))
    axarr = axarr.flatten()
    axarr[0].imshow(Image.open(BytesIO(image_a.content))) #replica_dir + 
    axarr[0].set_title(row["AuthorOriginal"].values[0] + '\n ' + row["Description"].values[0] + '\n REFERENCE')
    for i in range(len(sim_rerank[:11])):
        if sim[i] in list_theo:
            correct = 'CORRECT'
        else:
            correct = 'WRONG'

        row_2 = data[data['uid'] == sim_rerank[i]]
        info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n' + str(row_2["Description"].values[0])
    
        axarr[i+1].set_title(info_2 + '(rerank)\n' + 'Matches: ' + str(similarities_rerank[i]) + ', Grountruth: ' + correct)
        drawer = catch(sim_rerank[i], uid2path).split('/')[0]
        img = catch(sim_rerank[i], uid2path).split('_')[1].split('.')[0]
        image = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
        axarr[i+1].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.savefig('/scratch/students/schaerf/figures/' + row["uid"].values[0] + '_rerank.jpg')
    plt.show()
    
    f, axarr = plt.subplots(1,len(list_theo), figsize=(30,3))
    for i in range(len(list_theo)):
        axarr[i].set_title(str(i) + "th groudtruth image")
        corresp = catch(list_theo[i], uid2path)
        if type(corresp) == str:
            drawer = catch(list_theo[i], uid2path).split('/')[0]
            img = catch(list_theo[i], uid2path).split('_')[1].split('.')[0]
            image = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
            axarr[i].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.show()



    return sim[:20], sim_rerank[:20]

def show_results(row, embeddings, train_test, data, uid2path, fig_dir):
    
    tree, reverse_map = make_tree_orig(embeddings, True)

    if row["set"].values[0] in ['train', 'val', 'test']:
        list_theo = (
            list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
            + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
            + [row["uid"].values[0]]
        )
    else:
        raise Exception('File not in grountruth')
        
    sim, similarities = find_most_similar_orig(
        row["uid"].values[0], tree, embeddings, reverse_map, n=400+1, similarities=True
    )

    print('positions of matches', find_pos_matches(sim, list_theo, n=400))

    f, axarr = plt.subplots(3,4, figsize=(15,15))
    plt.suptitle('Image retrieval with embeddings of ' + row["Author"].values[0] + '\n ' + row["Description"].values[0] + 
                '\nRank: ' + ', '.join([str(x) for x in find_pos_matches(sim, list_theo, n=400)]))
    axarr = axarr.flatten()
    row_2 = data[data['uid'] == row["uid"].values[0]]
                              
    image_a = requests.get(get_drawer(row_2, row["uid"].values[0]))
        
    axarr[0].set_facecolor('black')
    axarr[0].imshow(Image.open(BytesIO(image_a.content))) #replica_dir + 
    axarr[0].set_title(row["Author"].values[0] + '\n ' + row["Description"].values[0] + '\n REFERENCE' + '\n ' + row["set"].values[0])
    for i in range(len(sim[:11])):
        if sim[i] in list_theo:
            correct = 'CORRECT'
        else:
            correct = 'NOT IN MORPHO'

        row_2 = data[data['uid'] == sim[i]]
        info_2 = str(row_2["Author"].values[0]) + '\n ' + str(row_2["Description"].values[0])
        info = info_2 + '\n' + 'Similarity: ' + str(np.round((1 - similarities[i]), 2)) + ', Grountruth: ' + correct 
        axarr[i+1].set_title("\n".join(wrap(info, 25)))
        image = requests.get(get_drawer(row_2, catch(sim[i], uid2path)))
        
        try:
            axarr[i+1].axis(False)
            axarr[i+1].set_aspect("equal")
            
            axarr[i+1].imshow(Image.open(BytesIO(image.content))) #replica_dir +

        except: 
            axarr[i+1].axis(False)
            axarr[i+1].set_aspect("equal")
            axarr[i+1].set_visible(False)
            
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(fig_dir + 'retrieval_' + row["Author"].values[0].split()[0] + '_' + row["Description"].values[0].split()[0] + '.jpg')
    plt.show()
    
    return sim[:20]

def show_clusters(cluster_num, morpho, uid2path, fig_dir, title):
    
    morpho_ = morpho.fillna('')
    new_morph = morpho_.groupby('uid').first().reset_index()
    cluster = new_morph[new_morph['cluster'] == cluster_num]

    f, axarr = plt.subplots(cluster.shape[0] // 4 + 1,4, figsize=(12,(cluster.shape[0] // 4 + 1) * 5))
    axarr = axarr.flatten()
    
    for i,row in cluster.reset_index().iterrows():
        
        if '2022' in str(row["annotated"]):
            info = str(row["Author"]) + '\n ' + str(row["Description"]) + '\n New Addition!'
        else:
            info = str(row["Author"]) + '\n ' + str(row["Description"]) + '\n ' + str(row["set"])
        axarr[i].set_title("\n".join(wrap(info, 30)), wrap=True)
        image = requests.get(get_drawer(pd.DataFrame(row), catch(row['uid'], uid2path)))
        
        try:
            axarr[i].axis(False)
            axarr[i].set_aspect("equal")
            axarr[i].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        except Exception as e:
            axarr[i].axis(False)
            axarr[i].set_aspect("equal")
            axarr[i].set_visible(False)
            print(get_drawer(pd.DataFrame(row), catch(row['uid'], uid2path)))
            continue

    for i in range(cluster.shape[0], (cluster.shape[0] // 4 + 1) * 4):
        axarr[i].axis(False)
        axarr[i].set_aspect("equal")
        axarr[i].set_visible(False)
    
    if cluster.shape[0] > 0:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(fig_dir + str(cluster_num) + '_' + row["Author"].split()[0] + '_' + row["Description"].split()[0] + '.jpg')
        plt.show()
    
def show_suggestions(row, embeddings, train_test, tree, reverse_map, uid2path, data):
    #replica_dir = '/mnt/project_replica/datasets/cini/'

    if row["set"].values[0] in ['train', 'val', 'test']:
        list_theo = (
            list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
            + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
            + [row["uid"].values[0]]
        )
    else:
        list_theo = [row["uid"].values[0]]
        
    sim = find_most_similar_no_theo(
        row["uid"].values[0], tree, embeddings, reverse_map, list_theo, n=8
    )

    f, axarr = plt.subplots(2,4, figsize=(30,10))
    axarr = axarr.flatten()
    drawer = row["path"].values[0].split('/')[0]
    img = row["path"].values[0].split('_')[1].split('.')[0]
    image_a = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
    axarr[0].imshow(Image.open(BytesIO(image_a.content))) #replica_dir + 
    axarr[0].set_title(row["AuthorOriginal"].values[0] + row["Description"].values[0])
    for i in range(len(sim)):
        row_2 = data[data['uid'] == sim[i]]
        info_2 = row_2["AuthorOriginal"].values[0] + ' ' + row_2["Description"].values[0]
        axarr[i+1].set_title(info_2)
        drawer = catch(sim[i], uid2path).split('/')[0]
        img = catch(sim[i], uid2path).split('_')[1].split('.')[0]
        image = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
        axarr[i+1].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.show()
    return row["uid"].values[0], sim
    



#########################################################
##### Model deprecated
#########################################################


def create_model(model_name, pooling):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_name == 'resnext-101':
        model = models.resnext101_32x8d(pretrained=True)
    elif model_name == 'regnet_x_32gf':
        model = models.regnet_y_32gf(pretrained=True)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
    elif model_name == "efficientnet0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "efficientnet7":
        model = models.efficientnet_b7(pretrained=True)

    if pooling == "avg":
        newmodel = torch.nn.Sequential(
            *(list(model.children())[:-2]), nn.AdaptiveAvgPool2d((1, 1))
        )
    elif pooling == 'max':
        newmodel = torch.nn.Sequential(
            *(list(model.children())[:-2]), nn.AdaptiveMaxPool2d((1, 1), )
        )
    
    return newmodel

def store_pretrained_embeddings(models, pools, resolutions):
    data_dir = '/scratch/students/schaerf/'
    data = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    for model in models: # ['resnext-101']'resnet50', 'efficientnet0', 'efficientnet7', 'resnet101', 'resnet152', 'densenet161', 'resnext-101', 'regnet_x_32gf', 
        for pool in pools: #'max', ['avg'] 
            for resolution in resolutions: #240, 480 [620]
                print(model, pool, resolution)
                newmodel = create_model(model, pool)
                embeddings = [[uid, get_embedding(preprocess_image(data_dir + 'subset/' + uid + '.jpg', resolution), newmodel).squeeze(1).squeeze(1)] for uid in tqdm(data['uid'].unique())]
                np.save(data_dir + 'models/' + model + '_' + pool + '_' + str(resolution) + '.npy', np.array(embeddings, dtype=np.ndarray))
    return 'done'


