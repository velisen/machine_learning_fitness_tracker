import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation 


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

pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns)+ 1), pca_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, number_comp=3)

subset = df_pca[df_pca['set'] == 35]
subset[['pca_1', 'pca_2', 'pca_3']].plot()
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z'] ** 2
gyr_r = df_squared['gyr_x'] ** 2 + df_squared['gyr_y'] ** 2 + df_squared['gyr_z'] ** 2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 14]
subset[['acc_r', 'gyr_r']].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns += ['acc_r', 'gyr_r']

ws =int(1000/ 200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(
        data_table=df_temporal,
        cols=[col],
        window_size=ws,
        aggregation_function='mean'
    )
    df_temporal = NumAbs.abstract_numerical(
        data_table=df_temporal,
        cols=[col],
        window_size=ws,
        aggregation_function='std'
    )
df_temporal_list = []
for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(
            data_table=subset,
            cols=[col],
            window_size=ws,
            aggregation_function='mean'
        )
        subset = NumAbs.abstract_numerical(
            data_table=subset,
            cols=[col],
            window_size=ws,
            aggregation_function='std')
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()
    
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
import os

# Print the current working directory
print("Current Directory:", os.getcwd())

# List all files and folders in the current directory
print("\nContents:")
for item in os.listdir():
    print(item)