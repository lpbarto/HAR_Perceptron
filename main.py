from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

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

# Apply the scaler to the feature training and test data
features_train_std = sc.transform(features_train)
features_test_std = sc.transform(features_test)

# Create a perceptron object and train
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
ppn.fit(features_train_std, target_train)

# Apply the trained perceptron on the features data to make predicts for the target test data
target_pred = ppn.predict(features_test_std)

labels=np.loadtxt(fname="UCI HAR Dataset/activity_labels.txt", dtype=str, usecols=(1))
cm= confusion_matrix(target_test, target_pred)
df = pd.DataFrame(cm, columns=labels, index=labels)
df.loc['Precision %',:]= df.sum(axis=0)
df.loc[:,'Recall %'] = df.sum(axis=1)

for label in labels:
    df.at["Precision %",label] = df.at[label,label] / df.at["Precision %",label] * 100
    df.at[label, "Recall %"] = df.at[label, label] / df.at[label, "Recall %"] * 100

df.at["Precision %", "Recall %"] = accuracy_score(target_test, target_pred) * 100
print(df)
