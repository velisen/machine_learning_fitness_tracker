import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_data_outliers_removed.pkl")

predictor_columns = list(df.columns[:6]) 

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]
duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    
    duration = stop - start
    df.loc[df['set'] == s, 'duration'] = duration.seconds
    
duration_df = df.groupby(['category'])['duration'].mean()
duration_df.iloc[0] / 5
duration_df.iloc[0] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.2

df_lowpass = LowPass.low_pass_filter(
    data_table=df_lowpass,
    col='acc_y',
    sampling_frequency=fs,
    cutoff_frequency=cutoff,
    order=5,
)

subset = df_lowpass[df_lowpass['set'] == 45]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='raw_data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='butterworth filter')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
