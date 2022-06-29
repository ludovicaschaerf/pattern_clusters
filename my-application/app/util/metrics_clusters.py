import pickle
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm 

from .utils import get_train_test_split, make_tree_orig, catch, find_most_similar_no_theo 


def update_morph(data_dir, morph_file, new=False):
    """_summary_

    Args:
        data_dir (_type_): _description_
        morph_file (_type_): _description_
        new (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    with open(data_dir + 'save_link_data_2018_08_02.pkl', 'rb') as f:
        morpho_graph_complete = pickle.load(f)
    morpho_graph_complete['cluster_file'] = 'Original'
    morpho_graph_complete['uid'] = ['_'.join(sorted([img1, img2])) for img1,img2 in zip(morpho_graph_complete['img1'], morpho_graph_complete['img2'])]
    
    metadata = pd.read_csv(data_dir + 'data.csv')
    metadata = metadata.drop(columns=['img1', 'img2', 'type', 'annotated', 'index', 'cluster', 'set', 'uid_connection', 'cluster_file'])

    ## function take what was already in train and test and preserve it (make train test split and then add the new ones)
    positives = get_train_test_split(metadata, morpho_graph_complete)
    
    if new:
        morpho_graph_clusters = pd.read_csv(data_dir + 'morphograph_clusters_new.csv')
    else:
        morpho_graph_clusters = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    morpho_graph_clusters['uid_connection'] = ['_'.join(sorted([img1, img2])) for img1,img2 in zip(morpho_graph_clusters['img1'], morpho_graph_clusters['img2'])]
    morpho_graph_clusters = morpho_graph_clusters.groupby(['uid_connection', 'type']).first().reset_index()
    if new:
        morpho_graph_clusters.to_csv(data_dir + 'morphograph_clusters_new.csv', index=False)
    else:
        morpho_graph_clusters.to_csv(data_dir + 'morphograph_clusters.csv', index=False)
    morpho_graph_clusters = morpho_graph_clusters[morpho_graph_clusters['cluster_file'].str.contains(morph_file)]
    morpho_graph_clusters['uid'] = morpho_graph_clusters['uid_connection']
    morpho_graph_clusters['annotated'] = morpho_graph_clusters['date']
    morpho_graph_clusters = morpho_graph_clusters.drop(columns=['uid_connection', 'date', 'cluster'])
    
    print('before adding the new ones', morpho_graph_complete[morpho_graph_complete['type'] == 'POSITIVE'].shape)
    morpho_graph_complete = pd.concat([morpho_graph_complete, morpho_graph_clusters], axis=0)
    print('after adding', morpho_graph_complete[morpho_graph_complete['type'] == 'POSITIVE'].shape)
    morpho_graph_complete = morpho_graph_complete.groupby('uid').first().reset_index()
    print('after deduplicating', morpho_graph_complete[morpho_graph_complete['type'] == 'POSITIVE'].shape)
    positives = get_new_split(metadata, positives, morpho_graph_complete)
    print(positives.shape)
    #positives = positives.groupby(['uid_connection']).first().reset_index()
    positives['old_cluster'] = positives['cluster']
    positives['cluster'] = positives['new_cluster']

    positives.to_csv(data_dir + 'morphograph/morpho_dataset.csv')
    
    return positives

def get_new_split(metadata, positives, morpho_update):
    """_summary_

    Args:
        metadata (_type_): _description_
        positives (_type_): _description_
        morpho_update (_type_): _description_

    Returns:
        _type_: _description_
    """    
    morpho_update = morpho_update[morpho_update["type"] == "POSITIVE"]
    morpho_update.columns = ["uid_connection", "img1", "img2", "type", "annotated", "cluster_file"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        morpho_update,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]
    
    # merging the two files
    positive = pd.concat(
        [
            positives,
            metadata.merge(morpho_update, left_on="uid", right_on="img1", how="inner"),
            metadata.merge(morpho_update, left_on="uid", right_on="img2", how="inner"),
        ],
        axis=0,
    )#.groupby(['uid', 'uid_connection']).first().reset_index()
    print(positive[positive['uid'] == positive['img1']].shape, positive[positive['uid'] == positive['img2']].shape,)
    positive = positive.groupby(['uid', 'uid_connection']).first().reset_index()
    print(positive[positive['uid'] == positive['img1']].shape, positive[positive['uid'] == positive['img2']].shape,)
    

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positive["new_cluster"] = positive["uid"].apply(lambda x: mapper[x])

    positive['set'] = positive['set'].fillna('no set')

    old2new = {idx:set for idx, set in zip(positive.groupby('new_cluster')['set'].max().index, positive.groupby('new_cluster')['set'].max().values)}
    positive['new set'] = positive['new_cluster'].apply(lambda x: old2new[x])
    positive.loc[positive['new set'] == 'no set', 'new set'] = 'train'

    return positive


def cluster_accuracy(cluster_annotations):
    """_summary_

    Args:
        cluster_annotations (_type_): _description_

    Returns:
        _type_: _description_
    """    
    cluster_info = cluster_annotations.groupby('cluster')['type'].apply(lambda x: x.value_counts()).reset_index()
    scores = {}
    for cluster, group in cluster_info.groupby('cluster'):
        if group.shape[0] == 1:
            if group['level_1'].values[0] in ['POSITIVE']:
                scores[cluster] = 1
            elif group['level_1'].values[0] in ['NEGATIVE']:
                scores[cluster] = 0
        elif 'POSITIVE' in group['level_1'].values:
            scores[cluster] = group[group['level_1'] == 'POSITIVE'].shape[0] / group.shape[0]
        
    return np.round(np.mean(list(scores.values())), 2)


def novelty_score(updated_morph, cluster_file, previous_cluster='Original'):
    """_summary_

    Args:
        updated_morph (_type_): _description_
        cluster_file (_type_): _description_
        previous_cluster (str, optional): _description_. Defaults to 'Original'.

    Returns:
        _type_: _description_
    """    
    before = updated_morph[updated_morph['cluster_file'] == previous_cluster]
    existing_clusters = before['new_cluster'].unique()
    after = updated_morph[updated_morph['cluster_file'] != previous_cluster][updated_morph['cluster_file'].str.contains(cluster_file)]
    additions = after[after['new_cluster'].isin(existing_clusters)]
    new_clusters = after[~after['new_cluster'].isin(existing_clusters)]

    scores = {
        'original size': before.shape[0],
        'newly added': after.shape[0],
        'additions to existing clusters': additions.shape[0],
        'number of clusters with new elements': additions['new_cluster'].nunique(),
        'new clusters' : new_clusters['new_cluster'].nunique(),
        'new clusters elements': new_clusters.shape[0],
        'progress': str(np.round(after.shape[0] / before.shape[0] * 100, 2)) + '%'

    }
    return scores


def evaluate_morph(positives, cluster_file, data_dir='../data/', set_splits=['train', 'val', 'test'], verbose=False):
    """_summary_

    Args:
        positives (_type_): _description_
        cluster_file (_type_): _description_
        data_dir (str, optional): _description_. Defaults to '../data/'.
        set_splits (list, optional): _description_. Defaults to ['train', 'val', 'test'].
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    with open(data_dir + cluster_file , 'rb') as infile:
        cluster_df = pickle.load(infile)
    
    
    #known = list(set(positives['img1'] + positives['img2']))
    #cluster_df = cluster_df[cluster_df['uid'].isin(known)]
    cluster_df = cluster_df[cluster_df['cluster'] != -1]
    if cluster_df[cluster_df['cluster'] != 0].shape[0] > 100:
        cluster_df = cluster_df[cluster_df['cluster'] != 0]
    scores_morph_recall = {}
    scores_morph_precision = {}
    for morph_cl, group_morph in positives.groupby('new_cluster'):
        if group_morph['set'].values[0] in set_splits:
            uids = list(set(group_morph['uid']))
            if len(uids) > 1:
                    
                if cluster_df[cluster_df['uid'].isin(uids)].shape[0] > 1:
                    max_cluster = cluster_df[cluster_df['uid'].isin(uids)].groupby('cluster').size().idxmax()
                    max_num = cluster_df[cluster_df['uid'].isin(uids)].groupby('cluster').size().max()
                    scores_morph_recall[morph_cl] = max_num/group_morph.shape[0] 
                    if cluster_df[cluster_df['cluster'] == max_cluster].shape[0] > 1:
                        scores_morph_precision[morph_cl] = (max_num - 1)/(cluster_df[cluster_df['cluster'] == max_cluster].shape[0]-1)
                    else:
                        scores_morph_precision[morph_cl] = 0
                else:
                    scores_morph_recall[morph_cl] = 0
                
    scores = {
        'cluster_file': cluster_file,
        'num_clusters': cluster_df['cluster'].nunique(),
        'num_clustered': cluster_df[cluster_df['cluster'] != -1].shape[0],
        
        'mean cluster precision': np.around(sum(list(scores_morph_precision.values())) / len(list(scores_morph_precision.values())), 2),
        'mean cluster recall' : np.around(sum(list(scores_morph_recall.values())) / len(list(scores_morph_recall.values())), 2),
    }
    if verbose:
        print(scores)
    return scores.values()



def make_new_train_set(embeddings, train_test, updated_morph, cluster_file, uid2path):
    'for each positive train with negative, if no negative, take closest one'
    """_summary_

    Returns:
        _type_: _description_
    """    
    after = updated_morph[updated_morph['cluster_file'].str.contains(cluster_file)]
    
    tree, reverse_map = make_tree_orig(embeddings, reverse_map=True)
    Cs = []
    for i in tqdm(range(train_test.shape[0])):
        list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
        if after.loc[after['img1'] == train_test["uid"][i], :].loc[after['type'] == 'NEGATIVE', :].shape[0] > 0:
            list_sim = list(after.loc[after['img1'] == train_test["uid"][i], :].loc[after['type'] == 'NEGATIVE', 'img2'].values)
            list_sim += find_most_similar_no_theo(
                train_test["uid"][i], tree, embeddings, reverse_map, list_theo, n=2
            )
        else:
            
            list_sim = find_most_similar_no_theo(
                train_test["uid"][i], tree, embeddings, reverse_map, list_theo, n=3
            )
            
        Cs.append(list_sim)
        
    train_test['C'] = Cs

    final = train_test[['img1', 'img2', 'C', 'new set']].explode('C')
    final.columns = ['A', 'B', 'C', 'set']
    final['A_path'] = final['A'].apply(lambda x: catch(x, uid2path))
    final['B_path'] = final['B'].apply(lambda x: catch(x, uid2path))
    final['C_path'] = final['C'].apply(lambda x: catch(x, uid2path))
    print(final.shape)

    final = final[final['C_path'].notnull() & final['A_path'].notnull() & final['B_path'].notnull()]
    print(final.shape)
    print(final.tail())

    return final 


def track_cluster_progression(cluster_annotations, cluster_file, previous_cluster_date, positives, data_dir='../data/', set_splits=['train', 'no set']):
    """_summary_

    Args:
        cluster_annotations (_type_): _description_
        cluster_file (_type_): _description_
        previous_cluster_date (_type_): _description_
        positives (_type_): _description_
        data_dir (str, optional): _description_. Defaults to '../data/'.
        set_splits (list, optional): _description_. Defaults to ['train', 'no set'].

    Returns:
        _type_: _description_
    """    
    with open(data_dir + cluster_file , 'rb') as infile:
        cluster_df = pickle.load(infile)
    
    cluster_annotations = cluster_annotations[cluster_annotations['cluster_file'].str.contains(previous_cluster_date)]
    print(cluster_annotations.shape)
    
    cluster_annotations = cluster_annotations[cluster_annotations['type'].isin(['POSITIVE', 'NEGATIVE'])].reset_index()

    print(cluster_annotations.shape)
    update_negatives = {}
    update_positives = {}
    for morph_cl, row in cluster_annotations.iterrows():
        img1 = row['img1']
        img2 = row['img2']
        if cluster_df[cluster_df['uid'] == img1]['cluster'].shape[0] > 0 and cluster_df[cluster_df['uid'] == img2]['cluster'].shape[0] > 0:
            if positives[positives['uid'] == img1]['set'].shape[0] > 0 and positives[positives['uid'] == img1]['set'].values[0] in set_splits:
                if row['type'] == 'POSITIVE':
                    if cluster_df[cluster_df['uid'] == img1]['cluster'].values[0] == cluster_df[cluster_df['uid'] == img2]['cluster'].values[0]:
                        update_positives[morph_cl] = 1
                    else:
                        update_positives[morph_cl] = 0
                if row['type'] == 'NEGATIVE':
                    if cluster_df[cluster_df['uid'] == img1]['cluster'].values[0] != cluster_df[cluster_df['uid'] == img2]['cluster'].values[0]:
                        update_negatives[morph_cl] = 1
                    else:
                        update_negatives[morph_cl] = 0
                
    #'check if negatives were correctly pushed away'
    scores = {
        'positives': sum(list(update_positives.values())) / len(list(update_positives.values())),
        'negatives' : sum(list(update_negatives.values())) / len(list(update_negatives.values())),
    }
    return scores
