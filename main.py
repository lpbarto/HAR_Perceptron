from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

FEATURE_TRAIN_PATH = "UCI HAR Dataset/train/X_train.txt"
TARGET_TRAIN_PATH = "UCI HAR Dataset/train/y_train.txt"
FEATURE_TEST_PATH = "UCI HAR Dataset/test/X_test.txt"
TARGET_TEST_PATH = "UCI HAR Dataset/test/y_test.txt"


feature_train = np.loadtxt(fname=FEATURE_TRAIN_PATH)
target_train = np.loadtxt(fname=TARGET_TRAIN_PATH)
feature_test = np.loadtxt(fname=FEATURE_TEST_PATH)
target_test = np.loadtxt(fname=TARGET_TEST_PATH)

