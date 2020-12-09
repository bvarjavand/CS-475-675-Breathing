import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pkl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight


def embed(X1_full, y1, window_size=5, stride=1, full_feat=False):
    X1 = X1_full["Volume"]
    samples = []
    labels = []
    for i in range(len(X1) // stride - window_size - 1):
        sample = X1[stride * i : stride * i + window_size]
        if full_feat:
            sample_len = len(sample)
            sample_cat = np.empty(sample_len + 2)
            sample_cat[:sample_len] = sample
            sample_cat[sample_len] = X1_full["from_thresh"][i]
            sample_cat[sample_len + 1] = X1_full["diff"][i]
            sample = sample_cat
        samples.append(sample)
        label = y1[stride * i + window_size]
        labels.append(label)
    return np.vstack(samples), np.vstack(labels)


def live_demo(X1_full, y1, window_size=20, stride=1, full_feat=False, embed_size=10000):
    X1 = X1_full["Volume"]
    if embed_size is None:
        embed_len = len(X1) // stride - window_size - 1
    else:
        embed_len = embed_size
    plotX = X1_full["Time"][window_size + 1 :]
    plotY = X1_full["Volume"][window_size + 1 :]
    preds = []

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.ion()

    cmap = ListedColormap(["k", "r", "g", "b"])
    norm = plt.Normalize(0, 3)
    ax.set_xlim(X1_full["Time"][:embed_len].min(), X1_full["Time"][:embed_len].max())
    ax.set_ylim(
        X1_full["Volume"][:embed_len].min() - 0.1,
        X1_full["Volume"][:embed_len].max() + 0.1,
    )

    # samples = []
    # labels = []
    preds = []
    for i in tqdm(range(embed_len)):
        # embed current point
        sample = X1[stride * i : stride * i + window_size]
        if full_feat:
            sample_len = len(sample)
            sample_cat = np.empty(sample_len + 2)
            sample_cat[:sample_len] = sample
            sample_cat[sample_len] = X1_full["from_thresh"][i]
            sample_cat[sample_len + 1] = X1_full["diff"][i]
            sample = sample_cat
        label = y1[stride * i + window_size]
        # predict/fit
        preds.append(clf.predict(sample.values.reshape(1, -1)))  # predict on new point
        clf.partial_fit(sample.values.reshape(1, -1), label.ravel())  # fit on new
        # samples.append(sample)
        # labels.append(label)
        if (i > 0) and (i % 10 == 0):
            # plot
            # points = np.array([plotX[:i], plotY[:i]]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # lc = LineCollection(segments, norm=norm)
            # lc.set_array(np.array(preds).ravel())
            # lc.set_linewidth(2)
            # line = ax.add_collection(lc)

            points = np.array([plotX[i - 11 : i], plotY[i - 11 : i]]).T.reshape(
                -1, 1, 2
            )
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, norm=norm)
            lc.set_array(np.array(preds[i - 11 : i]).ravel())
            lc.set_linewidth(2)
            line = ax.add_collection(lc)

            # plt.draw()
            plt.pause(0.00001)
    plt.ioff()
    plt.show()
    return preds


if __name__ == "__main__":
    fpaths = [f"./patients_data/patient_{i}/" for i in range(1, 13)]
    X = pd.DataFrame(pkl.load(open(fpaths[0] + "train_X.pkl", "rb")))
    y = pd.DataFrame(pkl.load(open(fpaths[0] + "train_y.pkl", "rb")))
    datasize = None  # DATA SIZE PARAM, ALL = None
    X = X[:datasize]
    y = y[:datasize]
    wsize = 10  # TUNING PARAM

    Xembed, yembed = embed(X, y["Label"], stride=1, window_size=wsize, full_feat=False)
    computed_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(yembed), y=yembed.flatten()
    )
    computed_weights_dict = dict.fromkeys(np.unique(yembed))
    for key in computed_weights_dict:
        computed_weights_dict[key] = computed_weights[key]

    print("Initial fit")
    clf = SGDClassifier(random_state=0).fit(Xembed, np.squeeze(yembed))
    #  class_weight=computed_weights_dict

    print("streaming...")
    i = 1
    X_session = pd.DataFrame(pkl.load(open(fpaths[0] + f"test_X{i}.pkl", "rb")))
    y_session = pd.DataFrame(pkl.load(open(fpaths[0] + f"test_y{i}.pkl", "rb")))
    live_demo(
        X_session,
        y_session["Label"].values.flatten(),
        window_size=wsize,
        embed_size=5000,
    )
