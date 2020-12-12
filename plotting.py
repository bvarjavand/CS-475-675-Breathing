import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pickle5 as pkl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

def get_plots(Xs, ys, preds, lim=1000, wsize=30):  #another size param. Lim=None for full plot
    for i in range(len(Xs)):
        xplot = Xs[i]['Time'][wsize+1:lim]
        yplot = Xs[i]['Volume'][wsize+1:lim]
        points = np.array([xplot, yplot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(1,1, figsize=(8,5))
        cmap = ListedColormap(['k', 'r', 'g', 'b'])
        norm = plt.Normalize(preds[i][:lim].min(), preds[i][:lim].max())
        lc = LineCollection(segments, norm=norm)
        lc.set_array(preds[i][:lim])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        ax.set_xlim(xplot.min(), xplot.max())
        ax.set_ylim(yplot.min()-.1, yplot.max()+.1)
        plt.show()


def get_gt(Xs, ys, lim=1000, wsize=30):
    xplot = Xs['Time'].values[wsize+1:lim]
    yplot = Xs['Volume'].values[wsize+1:lim]
    points = np.array([xplot, yplot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    cmap = ListedColormap(['y', 'b', 'k', 'r'])
    norm = plt.Normalize(ys["Label"].values[wsize+1:lim].min(), ys["Label"].values[wsize+1:lim].max())
    lc = LineCollection(segments, norm=norm)
    lc.set_array(ys["Label"].values[wsize+1:lim])
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    ax.set_xlim(xplot.min(), xplot.max())
    ax.set_ylim(yplot.min()-.1, yplot.max()+.1)


def plot_multiclass_roc(X_test, y_test, y_score, n_classes, figsize=(17, 6), title="default", wsize=30):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    for i in range(n_classes):
        y_test_oneVall = (y_test[wsize+1:]==i).astype(int)
        y_score_oneVall = (y_score==i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_test_oneVall, y_score_oneVall)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC {title}')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
