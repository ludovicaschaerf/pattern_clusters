from flask import Flask, render_template, request

from .util.utils_clusters import * 
from .util.metrics_clusters import update_morph
from .util.interest_metrics import add_interest_scores


app = Flask(__name__)

data_dir = "my-application/data/"
type = 'optics'

morpho = add_interest_scores(data_dir, translate=False, new=False, precomputed=True)
morpho_ = morpho.fillna('')
new_morph = morpho_.groupby('uid').first().reset_index()

def get_file_names(subfolder, eps):
        
    if subfolder == '07-06-2022/':
        data_file = 'data.csv'
    else:
        data_file = 'data_sample.csv' 
    
    data_norm = pd.read_csv(data_dir + data_file)
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
    
    clu2size = {i: cl for i,cl in zip(new_morph.groupby('cluster').size().index, new_morph.groupby('cluster').size().values)}
    new_morph['cluster_size'] = new_morph['cluster'].apply(lambda x: clu2size[x])
    new_morph_ = new_morph[new_morph['cluster_size'] > 1]
    
    INFO = images_in_clusters(new_morph_, morpho_, map_file=map_file, data_dir=data_dir)
       
    score_morph = {cluster: {col:group[col].values[0] for col in new_morph if 'scores' in col} for cluster, group in new_morph.groupby('cluster')}
    return render_template(
            "clusters.html",
            data=INFO,
            scores=score_morph,
            cold_start=request.method == "GET",
    )
    

# @app.route("/visual_clusters", methods=["GET", "POST"])
# def visual_clusters():
    
#     INFO = images_in_clusters(cluster_df, data_norm, map_file=map_file, data_dir=data_dir)
    
#     annotate_store(cluster_df, data_norm, map_file, cluster_file, data_dir)

#     return render_template(
#         "visual_clusters.html",
#         data=convert_to_json(INFO),
#         cold_start=request.method == "GET",
#     )


    
