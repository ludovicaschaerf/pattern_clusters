import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree


#########################################################
##### Embeddings
#########################################################

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
