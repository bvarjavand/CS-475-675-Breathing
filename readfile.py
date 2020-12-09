import glob
import pandas as pd
import numpy as np
import sys
import os
import sklearn
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    for fname in glob.glob("./test/*.pkl"):
        df = pd.read_pickle(fname)
 
        print(df)
        break

 
 
