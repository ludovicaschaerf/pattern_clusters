from flask import Flask, render_template, request

from .util.utils_clusters import * 
from .util.metrics_clusters import update_morph
from .util.interest_metrics import add_interest_scores


app = Flask(__name__)

data_dir = "my-application/data/"
type = 'optics'

# morphograph
# morpho = add_interest_scores(data_dir, translate=False, new=False)
# morpho.to_csv(data_dir + 'morphograph/morpho_dataset_enriched.csv')

morpho = pd.read_csv(data_dir + 'morphograph/morpho_dataset_enriched.csv')
# eps becomes number of clusters
# if type in ['mix','kmeans']:
#     eps = int(eps)

def get_file_names(subfolder, eps):
        
    data_file = 'data_sample.csv' 
    if subfolder == '28-05-2022/':
        data_file = subfolder + 'data_retrain_1.csv'
    if subfolder == '01-06-2022/':
        data_file = subfolder + 'data_retrain_2.csv'
    if subfolder == '07-06-2022/':
        data_file = 'data.csv'

    data_norm = pd.read_csv(data_dir + data_file)
    #embeds_file = subfolder + 'resnext-101_'+subfolder.strip('/') +'.npy' 
    map_file = subfolder + 'map2pos.pkl'
    cluster_file = subfolder + 'clusters_'+type+'_'+str(eps)+'_'+subfolder.strip('/')+'_19'
        
    with open(data_dir + cluster_file + '.pkl', 'rb') as infile:
        cluster_df = pickle.load(infile)
        cluster_df = cluster_df.sort_values('cluster')

    return data_norm, map_file, cluster_df, cluster_file


# clustering files
  
eps = 0.13
subfolder = '01-06-2022/'
data_norm, map_file, cluster_df, cluster_file = get_file_names(subfolder, eps)

eps_all = 0.13
subfolder_all = '07-06-2022/'
data_all, map_file_all, cluster_df_all, cluster_file_all = get_file_names(subfolder_all, eps_all)



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/clusters_subset", methods=["GET", "POST"])
def clusters_embeds():

    INFO, cluster = show_results_button(cluster_df, data_norm, map_file, data_dir=data_dir) 
    annotate_store(cluster_df, data_norm, map_file, cluster_file, data_dir) 
    
    return render_template(
        "clusters.html",
        item=cluster,
        data=INFO,
        cold_start=request.method == "GET",
    )

@app.route("/clusters", methods=["GET", "POST"])
def clusters_all():

    INFO, cluster = show_results_button(cluster_df_all, data_all, map_file, data_dir=data_dir) 
    annotate_store(cluster_df_all, data_all, map_file, cluster_file_all, data_dir) 
    
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
       
    score_morph = {cluster: {col:group[col].values[0] for col in new_morph if 'scores' in col} for cluster, group in new_morph.groupby('cluster')}
    return render_template(
        "clusters.html",
        data=INFO,
        scores=score_morph,
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


    
