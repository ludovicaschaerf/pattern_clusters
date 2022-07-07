# Semi-supervised Clustering of Visual Signatures of Artworks
## A human-in-the-loop approach to tracing visual pattern propagation in art history using deep computer vision methods.
## Code for the website served [here](https://patternclusters-api-heroku.herokuapp.com/). Master thesis by Ludovica Schaerf
 
 
This repository contains the full code and data files for serving the pattern-clusters website on Heroku.
The code from this repository can be run locally using the following commands:
```bash
git clone https://github.com/ludovicaschaerf/pattern_clusters.git
cd pattern_clusters
pip install -r requirements.txt
python ./my-application/wsgi.py
```


The repository contains:

```bash
├───my-application: containing all the code
│   ├───app: code and assets
│   │   ├───static: resources
│   │   │   ├───css
│   │   │   ├───images
│   │   │   ├───js
│   │   │   │   └───galleria-1.6.1
│   │   │   │       ├───dist
│   │   │   │       │   ├───plugins
│   │   │   │       │   │   ├───flickr
│   │   │   │       │   │   └───history
│   │   │   │       │   └───themes
│   │   │   │       │       ├───azur
│   │   │   │       │       ├───classic
│   │   │   │       │       ├───folio
│   │   │   │       │       ├───fullscreen
│   │   │   │       │       ├───miniml
│   │   │   │       │       └───twelve
│   │   │   │       └───src
│   │   │   │           ├───plugins
│   │   │   │           │   ├───flickr
│   │   │   │           │   └───history
│   │   │   │           └───themes
│   │   │   │               ├───azur
│   │   │   │               ├───classic
│   │   │   │               ├───folio
│   │   │   │               ├───fullscreen
│   │   │   │               ├───miniml
│   │   │   │               └───twelve
│   │   │   ├───scss
│   │   │   └───vendor
│   │   │       ├───bootstrap
│   │   │       │   ├───css
│   │   │       │   └───js
│   │   │       └───jquery
│   │   ├───templates: templates for serving pages, contains base navigation header, cluster file, landing page
│   │   ├───util: utility files containing python functions 
│   │   │   └───__pycache__
|   |   ├───main.py: python file serving all flask pages.
│   │   └───__pycache__

│   └───data
│       ├───01-06-2022: subset data, containing: 2d positions file, cluster file, metadata
│       ├───07-06-2022: full data
│       └───morphograph: morphograph data
└───__pycache__ 
```
