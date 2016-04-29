from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import auc, accuracy_score
import pickle

import pandas as pd
import numpy as np
import pickle
import json
import time




def main():

    # read bag of words with 5000 features
    X = np.load("data/bagOfWods_5000.npz")['X'][()]
    Y = np.load("data/bagOfWods_5000.npz")['y'][()]
    
    # split into subimssion dataset and training dataset
    test_X = X[0:50000, :]
    test_Y = Y[0:50000]
    train_X = X[50000:, :]
    train_Y = Y[50000:]

    # split dataset intro training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size= 0.2, random_state= 0)

    #  set grid search parameters
    xgb_params = {'max_depth' : [11], 'learning_rate':[0.1],
                  'gamma':[0], 'n_estimators': [4000],
                  "objective": ["binary:logistic"], 'nthread' : [36]}

    xgb = XGBClassifier()

    # gridsearch cv with 11 fold
    grid = GridSearchCV(xgb, xgb_params, scoring='roc_auc', cv = 11)

    print("running model....")

    start = time.time()

    # fitting model to training data.
    grid.fit(X_train, y_train)

    # save completed model to pickle
    output = open("cache/xgboost_grid.pkl", 'wb')
    pickle.dump(grid, output)
    output.close()

    score = grid.best_score_
    best_estimator = grid.best_estimator_
    best_params = grid.best_params_
    result = grid.predict(X_test)
    accuracy = accuracy_score(y_test, result)

    # aggregate result to dictionary
    report = {
        "best_estimator" : best_estimator,
        "best_score" : score,
        "best_params" : best_params,
        "accuracy" : accuracy
    }
    report_str = str(report)

    # save the report as a json format
    with open("report/report.json", "w") as outfile:
        json.dump(report_str, outfile)

    # make a submission file
    test_result = grid.predict(test_X)
    submission = pd.DataFrame({'id': np.arange(1, 50001), 'y': test_result})
    submission.to_csv("submission/filename.csv", index = False)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()


