import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import pickle5 as pkl


def embed(X1_full, y1, window_size = 5, stride = 1, full_feat=True):
    X1 = X1_full['Volume']
    samples = []
    labels = []
    for i in range(len(X1)//stride-window_size-1):
        sample = X1[stride*i:stride*i + window_size]
        if full_feat:
            sample_len = len(sample)
            sample_cat = np.empty(sample_len + 2)
            sample_cat[:sample_len] = sample
            sample_cat[sample_len] = X1_full['from_thresh'][i]
            sample_cat[sample_len + 1] = X1_full['diff'][i]
            sample = sample_cat
        samples.append(sample)
        label = y1[stride*i + window_size]
        labels.append(label)
    return np.vstack(samples), np.vstack(labels)


def get_preds(clf, num, fpaths, wsize, dsize=None, full_feat=True):  # another data size parameter. All = None
    Xs = []
    ys = []
    preds = []
    for i in range(1,num+1):
        X_session = pd.DataFrame(pkl.load(open(fpaths[0]+f"test_X{i}.pkl", "rb")))
        y_session = pd.DataFrame(pkl.load(open(fpaths[0]+f"test_y{i}.pkl", "rb")))
        X_session = X_session[:dsize]
        y_session = y_session[:dsize]
        X_s_test, y_s_test =  embed(
            X_session, y_session['Label'], stride=1, window_size=wsize, full_feat=full_feat)
        pred = np.zeros(len(y_s_test))
        for online_idx in tqdm(range(len(y_s_test))):
            pred[online_idx] = clf.predict(X_s_test[online_idx].reshape(1,-1))  # predict on new point
            clf.partial_fit(X_s_test[online_idx].reshape(1, -1), y_s_test[online_idx])  # fit on new
        Xs.append(X_session)
        ys.append(y_session)
        preds.append(pred)
    return Xs, ys, preds


def print_stats(y_test, preds):
    print("MSE:", mean_squared_error(y_test, preds))
    print("acc:", accuracy_score(y_test, preds))
    class_acc_list = []
    for label in np.unique(y_test):
        pred_idx = np.where(preds == label)
        class_acc = accuracy_score(y_test[pred_idx], preds[pred_idx])
        class_acc_list.append(class_acc)
        print(f"class {label} acc:", class_acc)
    print("avg class acc (unweighted):", sum(class_acc_list) / len(class_acc_list))
