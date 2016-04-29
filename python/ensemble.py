from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, accuracy_score
import pickle
import time
import pandas as pd
import numpy as np
import pickle
import json
import time


def main():

    # load data and split into submission and training data
    X = np.load("data/bagOfWods_3000.npz")['X'][()]
    Y = np.load("data/bagOfWods_3000.npz")['y'][()]
    test_X = X[0:50000, :]
    # test_Y = Y[0:50000]
    train_X = X[50000:, :]
    train_Y = Y[50000:]



    # split into train/test
    X_train, X_temp, y_train, y_temp = train_test_split(train_X, train_Y, test_size= 0.3, random_state= 0)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1)

    print("running model....")

    name_report = "report/report_%s.json" % "ensemble"

    with open(name_report, mode='w') as f:
        json.dump([], f)
    with open(name_report, mode='r') as modeljson:
        models = json.load(modeljson)



    start = time.time()

    # dictionary of different models with their parameters
    model_dic = {

        "randomForest" : RandomForestClassifier(n_jobs=-1, n_estimators= 3000, max_depth=10),
        "logistic" : LogisticRegression(n_jobs=-1),
        "svmrdf": SVC(probability=True),
        "linearSVM": LinearSVC(),
        "extra" : ExtraTreesClassifier(n_estimators=3000, max_depth=10, n_jobs=-1)
    }

    # parameter grid
    PARAM_GRID = {
        "randomForest" : {
            'n_estimators': [2000],
            'max_depth' : [8, 11]

        },
        "logistic" : {

        },
        "svmrdf" : {
            'C' : [0.1, 1]
        },
        "linearSVM" : {

        },
        "extra" : {
            'n_estimators' : [2000],
            'max_depth' : [8, 11]
        }

    }


    # loop through dictionary of models and fit the model on data
    for model_name, grid in model_dic.items():

        print("now %s is running" % model_name)
        print(PARAM_GRID[model_name])


        # grid = GridSearchCV(model, PARAM_GRID[model_name], scoring='accuracy', n_jobs=-1, cv=7)

        grid.fit(X_train, y_train)

        # output model
        model_file_name = "cache/%s.pkl" %(model_name)
        output = open(model_file_name, 'wb')
        pickle.dump(grid, output)
        output.close()


        if model_name == 'gbm' or model_name == 'svmrdf':
            result = grid.predict(X_test.toarray())
            result_prob = grid.predict_proba(X_test.toarray())
        else:
            result = grid.predict(X_test)

        accuracy = accuracy_score(y_test, result)

        end = time.time()
        time_delay = end - start

        submission_name = "submission/submission_file_%s.csv" %(model_name)
        report = {
            "model_name" : model_name,
            "accuracy" : accuracy,
            "time_delay" : time_delay,
            "submission_name" : submission_name
        }
        report_str = str(report)

        with open(name_report, mode='w') as modeljson:

            models.append(report_str)
            json.dump(models, modeljson)

        if model_name == 'gbm' or model_name == 'svmrdf':
            test_result = grid.predict(test_X.toarray())
        else:
            test_result = grid.predict(test_X)

        # make a submission file.
        submission = pd.DataFrame({'id': np.arange(1, 50001), 'y': test_result})
        submission.to_csv(submission_name, index = False)


if __name__ == '__main__':
    main()


