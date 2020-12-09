from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import glob
import pandas as pd

import statsmodels.tsa.api as smt

files = glob.glob("./test/*.pkl")
y = pd.read_pickle(files[0])['Volume']


def plot_multi_acf(data, lags, titles, ylim=None, partial=False):
    num_plots = len(lags)
    fig, ax = plt.subplots(len(lags), 1, figsize=(10, 3 * num_plots))
    if num_plots == 1:
        ax = [ax]
    acf_func = smt.graphics.plot_pacf if partial else smt.graphics.plot_acf
    for idx, (lag, title) in enumerate(zip(lags, titles)):
        fig = acf_func(data, lags=lag, ax=ax[idx], title=title)
        if ylim is not None:
            ax[idx].set_ylim(ylim)

    fig.tight_layout()

period_minutes = 5
samples_per_hour = int(60 / period_minutes)
samples_per_day = int(24 * samples_per_hour)
samples_per_week = int(7 * samples_per_day)

lags = [3 * samples_per_hour]  # , samples_per_day, samples_per_week
titles= ['Autocorrelation: 3-Hour Lag',
         'Autocorrelation: 1-Day Lag',
         'Autocorrelation: 1-Week Lag']

plot_multi_acf(y, lags, titles)
 
# files = glob.glob("./test/*.pkl")
# for fname in files:
#     df = pd.read_pickle(fname)
#     Xtemp = df["Volume"]
#     ytemp = df["Label"]
#     # fit model
#     model = ARIMA(Xtemp, order=(5,1,0))
#     model_fit = model.fit(disp=0)
#     # plot residual errors
#     residuals = pd.DataFrame(model_fit.resid)
#     residuals.plot()
#     pyplot.plot(Xtemp, alpha=0.5)
#     pyplot.show()
