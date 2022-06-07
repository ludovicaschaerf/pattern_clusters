import pickle
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm 

from python_files.utils import get_train_test_split, make_tree_orig, catch, find_most_similar_no_theo 

def update_morph(data_dir, morph_file):
    with open(data_dir + 'save_link_data_2018_08_02.pkl', 'rb') as f:
        morpho_graph_complete = pickle.load(f)
    morpho_graph_complete['cluster_file'] = 'Original'
    
    metadata = pd.read_csv(data_dir + 'data_sample.csv')
    metadata = metadata.drop(columns=['img1', 'img2', 'type', 'annotated', 'index', 'cluster', 'set', 'uid_connection'])

    ## function take what was already in train and test and preserve it (make train test split and then add the new ones)
    positives = get_train_test_split(metadata, morpho_graph_complete)
    
    morpho_graph_clusters = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    morpho_graph_clusters = morpho_graph_clusters.groupby(['img1', 'img2', 'type']).first().reset_index()
    morpho_graph_clusters.to_csv(data_dir + 'morphograph_clusters.csv', index=False)
    
    morpho_graph_clusters = morpho_graph_clusters[morpho_graph_clusters['cluster_file'].str.contains(morph_file)]
    morpho_graph_clusters['uid'] = morpho_graph_clusters['uid_connection']
    morpho_graph_clusters['annotated'] = morpho_graph_clusters['date']
    morpho_graph_clusters = morpho_graph_clusters.drop(columns=['uid_connection', 'date', 'cluster'])
    morpho_graph_complete = pd.concat([morpho_graph_complete, morpho_graph_clusters], axis=0)
    
    positives = get_new_split(metadata, positives, morpho_graph_complete)

    positives = positives.groupby('uid_connection').first().reset_index()
    #positive = positives.groupby('uid').last().reset_index()
    positives['old_cluster'] = positives['cluster']
    positives['cluster'] = positives['new_cluster']
    positives.to_csv(data_dir + 'morphograph/morpho_dataset.csv')
    
    return positives


def cluster_accuracy(cluster_annotations):
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
    before = updated_morph[updated_morph['cluster_file'] == previous_cluster]
    existing_clusters = before['new_cluster'].unique()
    after = updated_morph[updated_morph['cluster_file'].str.contains(cluster_file)]
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


def evaluate_morph(positives, cluster_file, data_dir='../data/', set_splits=['train', 'val', 'test']):
    with open(data_dir + cluster_file , 'rb') as infile:
        cluster_df = pickle.load(infile)
    
    
    #known = list(set(positives['img1'] + positives['img2']))
    #cluster_df = cluster_df[cluster_df['uid'].isin(known)]

    scores_morph_recall = {}
    scores_morph_precision = {}
    for morph_cl, group_morph in positives.groupby('new_cluster'):
        if group_morph['set'].values[0] in set_splits:
            uids = list(set(group_morph['uid']))
            if cluster_df[cluster_df['uid'].isin(uids)].shape[0] > 0:
                max_num = cluster_df[cluster_df['uid'].isin(uids)].groupby('cluster').size().values[0]
                scores_morph_recall[morph_cl] = max_num/group_morph.shape[0]
                scores_morph_precision[morph_cl] = max_num/cluster_df[cluster_df['uid'].isin(uids)].shape[0]

    scores = {
        'precision': sum(list(scores_morph_precision.values())) / len(list(scores_morph_precision.values())),
        'recall' : sum(list(scores_morph_recall.values())) / len(list(scores_morph_recall.values())),
    }
    return scores


def get_new_split(metadata, positives, morpho_update):

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
    ).groupby('uid_connection').first().reset_index()

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positive["new_cluster"] = positive["uid"].apply(lambda x: mapper[x])

    positive['set'] = positive['set'].fillna('no set')

    old2new = {idx:set for idx, set in zip(positive.groupby('new_cluster')['set'].max().index, positive.groupby('new_cluster')['set'].max().values)}
    positive['new set'] = positive['new_cluster'].apply(lambda x: old2new[x])
    positive.loc[positive['new set'] == 'no set', 'new set'] = 'train'

    return positive

def make_new_train_set(embeddings, train_test, updated_morph, cluster_file, uid2path):
    'for each positive train with negative, if no negative, take closest one'
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
