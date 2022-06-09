from app.main import app

#heroku ps:copy my-application/data/morphograph_clusters.csv --app patternclusters-api-heroku  
if __name__ == "__main__":
        app.run(port=8080, debug=True)