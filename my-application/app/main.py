from flask import Flask, render_template, request
import argparse

from .util.utils_clusters import * 
from .util.metrics_clusters import update_morph


app = Flask(__name__)

# parser = argparse.ArgumentParser(description='Model specifics')

# parser.add_argument('--data_dir', dest='data_dir',
#                         type=str, help='', default="../data/")
# parser.add_argument('--subfolder', dest='subfolder',
#                         type=str, help='', default="01-06-2022/")
# parser.add_argument('--precomputed', dest='precomputed',
#                         type=bool, help='', default=True)
# parser.add_argument('--type', dest='type',
#                         type=str, help='', default='dbscan')
# parser.add_argument('--eps', dest='eps',
#                         type=float, help='', default=0.08)

# args = parser.parse_args()

data_dir = "data/"
type = 'dbscan'
eps = 0.08
subfolder = '01-06-2022/'

# morphograph
# update_morph(args.data_dir, '-2022')

morpho = pd.read_csv(data_dir + 'morphograph/morpho_dataset.csv')

# eps becomes number of clusters
if type in ['mix','kmeans']:
    eps = int(eps)

    
# clustering files
data_file = 'data_sample.csv' 
if subfolder == '28-05-2022/':
    data_file = subfolder + 'data_retrain_1.csv'
if subfolder == '01-06-2022/':
    data_file = subfolder + 'data_retrain_2.csv'

data_norm = pd.read_csv(data_dir + data_file)
# embeds_file = args.subfolder + 'resnext-101_'+args.subfolder.strip('/') +'.npy' 
map_file = subfolder + 'map2pos.pkl'
cluster_file = subfolder + 'clusters_'+type+'_'+str(eps)+'_'+subfolder.strip('/')+'_19'
    
with open(data_dir + cluster_file + '.pkl', 'rb') as infile:
    cluster_df = pickle.load(infile)
    cluster_df = cluster_df.sort_values('cluster')



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/clusters_embeds", methods=["GET", "POST"])
def clusters_embeds():

    INFO, cluster = show_results_button(cluster_df, data_norm, map_file, data_dir=data_dir) 
    annotate_store(cluster_df, data_norm, map_file, cluster_file, data_dir) 
    
    return render_template(
        "clusters.html",
        item=cluster,
        data=INFO,
        cold_start=request.method == "GET",
    )


@app.route("/morphograph", methods=["GET", "POST"])
def morpho_show():
    
    new_morph = morpho.groupby('uid').first().reset_index()
    print('morph clusters')
    print(new_morph['cluster'].nunique())
    clu2size = {i: cl for i,cl in zip(new_morph.groupby('cluster').size().index, new_morph.groupby('cluster').size().values)}
    new_morph['cluster_size'] = new_morph['cluster'].apply(lambda x: clu2size[x])
    new_morph = new_morph[new_morph['cluster_size']>1]
    print(new_morph['cluster'].nunique())
    
    INFO = images_in_clusters(new_morph, morpho, map_file=map_file, data_dir=data_dir)
        
    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )


@app.route("/visual_clusters", methods=["GET", "POST"])
def visual_clusters():
    
    INFO = images_in_clusters(cluster_df, data_norm, map_file=map_file, data_dir=data_dir)
    
    annotate_store(cluster_df, data_norm, map_file, cluster_file, data_dir)

    return render_template(
        "visual_clusters.html",
        data=convert_to_json(INFO),
        cold_start=request.method == "GET",
    )


    
