# Disclaimer

I do not own the right of those notebooks, they were made by Jliporace. https://github.com/Jliporace

## H2O Tutorial Files

The main goal of this project was to explore the funcionalities of H2O'3 AutoML. In the process, some tutorials files were build in Jupyter Notebook and H2O's flow and may serve as a first orientation to those who also intend to use h2o's automl. 
One of the products of this project was a presentation, that can be found at:. A more detailed version of the notebooks and flows used in the presentation can be found here. 

In those files, you shall find brief explanations about the H2O's AutoML process as well as some coding instructions, described below: 

1) Jupyter Notebook 

- How to connect to H2O's Hurb cluster
- How to import/export and save files 
- How to train and save AutoML models in the cluster and in GCS
- How to use generated/saved models to predict results in new datasets
- How to perform checkpoiting: train new models using previous models as a base

2) H2O's flow (flows directory)
- How to import files from GCS
- How to parse files
- How to split frames into training/testing frames
- How to build an AutoML run
- How to use the generated with AutoML to run new predictions
