#
#  Lapo Bartolacci
#

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def main():
    FEATURE_TRAIN_PATH = "UCI HAR Dataset/train/X_train.txt"
    TARGET_TRAIN_PATH = "UCI HAR Dataset/train/y_train.txt"
    FEATURE_TEST_PATH = "UCI HAR Dataset/test/X_test.txt"
    TARGET_TEST_PATH = "UCI HAR Dataset/test/y_test.txt"

    features_train = np.loadtxt(fname=FEATURE_TRAIN_PATH)
    target_train = np.loadtxt(fname=TARGET_TRAIN_PATH)
    features_test = np.loadtxt(fname=FEATURE_TEST_PATH)
    target_test = np.loadtxt(fname=TARGET_TEST_PATH)

    sc = StandardScaler()
    sc.fit(features_train)
    features_train_std = sc.transform(features_train)
    features_test_std = sc.transform(features_test)

    useGridSearchCV(features_train_std, target_train, features_test_std, target_test)


def useGridSearchCV(features_train, target_train, features_test, target_test):
    param_grid = [
        {'penalty': ['l2', 'l1', 'elasticnet', None], 'alpha': [0.001, 0.0001, 0.00001],
         'max_iter': [100, 1000, 10000],
         'tol': [1e-3, 1e-4, 1e-2, 1e-3], 'n_jobs': [-1], 'eta0': [1, 0.1], 'random_state': [0, None]}
    ]
    ppn = Perceptron()
    grid_search = GridSearchCV(ppn, param_grid, cv=5, return_train_score=True, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(features_train, target_train)
    print("parametri migliori")
    print(grid_search.best_params_)
    print("Score migliore")
    print(grid_search.best_score_)
    bestPred = grid_search.predict(features_test)
    makeConfusionMatrix(target_test, bestPred)


def makeConfusionMatrix(Y_test, Y_pred):
    labels = np.loadtxt(fname="UCI HAR Dataset/activity_labels.txt", dtype=str, usecols=(1))
    cm = confusion_matrix(Y_test, Y_pred, labels=[1, 2, 3, 5, 4, 6])
    df = pd.DataFrame(cm, columns=labels, index=labels)
    df.loc['Precision %', :] = df.sum(axis=0)
    df.loc[:, 'Recall %'] = df.sum(axis=1)

    for label in labels:
        df.at["Precision %", label] = (df.at[label, label] / df.at["Precision %", label] * 100).astype(float).round(2)
        df.at[label, "Recall %"] = (df.at[label, label] / df.at[label, "Recall %"] * 100).astype(float).round(2)

    df.at["Precision %", "Recall %"] = (accuracy_score(Y_test, Y_pred) * 100).round(2)

    pd.options.display.width = 0
    df.to_csv("confusion_matrix.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


if __name__ == "__main__":
    main()



