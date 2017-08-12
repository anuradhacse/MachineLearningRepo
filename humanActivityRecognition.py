import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Cython.Shadow import inline
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('/home/anuradha/PycharmProjects/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
print(df.head())

df['activity'].value_counts().plot(kind='bar', title='Training examples by activity type')

# plt.show()
# plt.savefig( 'myfig.png' )

df['user'].value_counts().plot(kind='bar', title='Training examples by user');
# plt.show()

def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

# plot_activity("Sitting", df)
plot_activity("Jogging", df)

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

print(np.array(segments).shape)

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

print(reshaped_segments.shape)