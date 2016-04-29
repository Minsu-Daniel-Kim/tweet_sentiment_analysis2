
our basic implemention is most done in index.py using XGBoost. index.py is a slight modified version of index.py in that it loops through datasets with different number of features. Most of parameters are predetermined by our experimentations on R. 

If you look at ensemble.py, it has a dictionary of different models. We ran a gridsearch on all of the models in dictionary.

Whenever a model is run, the model is saved in pkl file and its accuracy is save in json foramt. Also, we generate submission file for each model since running a single model takes a long time.