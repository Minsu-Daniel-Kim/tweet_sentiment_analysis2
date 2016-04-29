from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import datasets
from xgboost import XGBClassifier
import xgboost
from sklearn.metrics import auc, accuracy_score
import pickle
import time
import pandas as pd
import numpy as np
import pickle
import json
import time


def main():
    print("Let's start!")

    # loop through datasets with different number of features
    for features in [5000, 6000, 7000, 8000, 9000, 10000, 4000, 3000]:
        modeling(features)

# modeling
def modeling(features):

    # read a bag of words
    bag_file_name = "data/bagOfWods_%s.npz" %features

    X = np.load(bag_file_name)['X'][()]
    Y = np.load(bag_file_name)['y'][()]

    # split dataset into submission and training
    test_X = X[0:50000, :]
    train_X = X[50000:, :]
    train_Y = Y[50000:]

    # split into training /testing
    X_train, X_temp, y_train, y_temp = train_test_split(train_X, train_Y, test_size= 0.3, random_state= 0)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)

    print("running model....")

    # xgboost matrix
    dtrain = xgboost.DMatrix(X_train, label=y_train)


    num_round = 6000
    n_fold = 12

    name_report = "report/report_%s.json" %features

    with open(name_report, mode='w') as f:
        json.dump([], f)
    with open(name_report, mode='r') as modeljson:
        models = json.load(modeljson)

    # loop through tree depth of 13 and 15
    for depth in [13, 15]:

        start = time.time()

        # xgboost paramater
        param = {'bst:max_depth': depth, 'bst:eta' : 0.05, 'silent' : 1, 'objective' : 'binary:logistic' }
        param['nthread'] = 36
        param['eval_metric'] = 'error'

        # run xgboost cross validation
        result_cv = xgboost.cv(param, dtrain, num_round, nfold=n_fold,seed = 0)
        
        # minumum number of rounds
        nround = np.asscalar(np.argwhere(result_cv["test-error-mean"] == np.min(result_cv["test-error-mean"]))[0])

        # xgboost parameters
        xgb_params = {'max_depth' : [depth], 'learning_rate':[0.05],
                      'n_estimators': [nround], "objective": ["binary:logistic"], 'nthread' : [36]}

        xgb = XGBClassifier()

        # gridsearchcv
        grid = GridSearchCV(xgb, xgb_params, scoring='accuracy')
        grid.fit(X_train, y_train)

        # output model
        model_name = "cache/xgboost_depth_%s_nround_%s_features_%s.pkl" %(depth, nround, features)
        output = open(model_name, 'wb')
        pickle.dump(grid, output)
        output.close()


        best_params = grid.best_params_
        result = grid.predict(X_test)
        accuracy = accuracy_score(y_test, result)

        end = time.time()
        time_delay = end - start

        submission_name = "submission/submission_file_depth_%s_nround_%s_features_%s.csv" %(depth, nround, features)

        report = {
            "model_name" : model_name,
            "accuracy" : accuracy,
            "time_delay" : time_delay,
            "best_params" : best_params,
            "submission_name" : submission_name
        }
        report_str = str(report)

        # save result as a json format
        with open(name_report, mode='w') as modeljson:

            models.append(report_str)
            json.dump(models, modeljson)

        # make a submission file
        test_result = grid.predict(test_X)
        submission = pd.DataFrame({'id': np.arange(1, 50001), 'y': test_result})
        submission.to_csv(submission_name, index = False)


if __name__ == '__main__':
    main()


