from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import glob
import pandas as pd
import numpy as np

train_files = glob.glob("./train/*.pkl")
train_df = pd.read_pickle(train_files[0])
train_X = train_df[["Time", "Volume", "from_thresh", "diff","ID"]]
train_y = train_df["Label"]

test_files = glob.glob("./test/*.pkl")
test_df = pd.read_pickle(test_files[0])
test_X = test_df[["Time", "Volume", "from_thresh", "diff","ID"]]
test_y = test_df["Label"]

def add_df(X, y, fname):
    df = pd.read_pickle(fname)
    Xtemp = df[["Time", "Volume", "from_thresh", "diff","ID"]]
    ytemp = df["Label"]
    return X.append(Xtemp), y.append(ytemp)
    
for fname in train_files[1:]:
    X_train, y_train = add_df(train_X, train_y, fname)

for fname in test_files[1:]:
    X_test, y_test = add_df(test_X, test_y, fname)

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

np.save('./dataset/train.feats.npy', X_train_np)
np.save('./dataset/train.labels.npy', y_train_np)
np.save('./dataset/dev.feats.npy', X_test_np)
np.save('./dataset/dev.labels.npy', y_test_np) 

# X_train.to_pickle("./dataset/X_train.pkl")
# y_train.to_pickle("./dataset/y_train.pkl")
# X_test.to_pickle("./dataset/X_test.pkl")
# y_test.to_pickle("./dataset/y_test.pkl")
