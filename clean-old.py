import glob
import pandas as pd
import numpy as np
import sys
import os

def get_labels(idx, label=1):
    """ Gets labels by filling in until a min in reached.
    """
    min = df['Volume'][idx]
    while 1:
        if label==1:
            idx -= 1
        elif label==3:
            idx += 1
        curr_val = df['Volume'][idx]
        if curr_val > min:
            return
        else:
            df.at[idx, 'Label'] = label
            min = curr_val

if __name__ == "__main__":
    user_input = input("Enter the path of your file: ")
    assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
    # OneDrive_2_11-12-2020
    for fname in glob.glob("./"+user_input+"/*.dat"):
        f=open(fname)
        lines=f.readlines()
        df = pd.read_csv(fname,
                    sep=";",
                    skiprows=26,
                    usecols=[0,1],
                    names=['Time','Volume'])
        df.attrs["threshold"] = lines[2].split(":")[1]
        df['Label'] = (df['Volume']>float(df.attrs["threshold"])).astype(int)*2
        # get edges
        edges = df['Label'].shift(-1, fill_value=0)-df['Label']
        idx1 = np.where(edges>0)[0]
        idx3 = np.where(edges<0)[0][:-2]
        for left_edge_idx in idx1:
            get_labels(left_edge_idx, label=1)
        for right_edge_idx in idx3:
            get_labels(right_edge_idx, label=3)
        df.to_pickle("./test/"+fname.split('_')[-1].split('.')[0]+".pkl")
