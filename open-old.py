import glob
import pandas as pd
import numpy as np
files = glob.glob("./test/*.pkl")
df = pd.read_pickle(files[0])
X = df[["Time", "Volume"]]
y = df["Label"]

def add_df(X, y, fname):
    df = pd.read_pickle(fname)
    Xtemp = df[["Time", "Volume"]]  # , "from_thresh", "diff"
    ytemp = df["Label"]
    return X.append(Xtemp), y.append(ytemp)
    
for fname in files[1:]:
    X, y = add_df(X, y, fname)

X.to_pickle("X.pkl")
y.to_pickle("y.pkl")