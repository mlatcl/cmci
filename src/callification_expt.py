
import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from umap import UMAP
from librosa.feature import mfcc
from librosa.core.spectrum import stft
from scipy.io import wavfile as wav
from scipy.signal import spectrogram

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

DATA_LOC = '../data/Calls For ML/'

def process(f, start, end):
    a = read_audio(f + '.wav')[int(start * sr):int(end * sr)]
    # S = spectrogram(a, nperseg=len(a)//3, noverlap=len(a)//12, fs=sr)[-1]
    S = np.abs(stft(a, n_fft=len(a)//3, hop_length=len(a)//6))
    features = mfcc(S=S, n_mfcc=20)
    features = (features - features.mean()) / (features.std() + 1e-6)
    return features.reshape(-1)

@lru_cache(maxsize=10)
def read_audio(f):
    return wav.read(os.path.join(DATA_LOC, f))[1].mean(axis=1)

if __name__ == '__main__':

    calls = pd.read_excel(os.path.join(DATA_LOC, 'Calls_ML.xlsx'))
    calls.columns = [c.lower().replace(' ', '_') for c in calls.columns]
    calls = calls.loc[calls.start < calls.end].reset_index(drop=True)
    calls['call_type'] = calls.call_type.apply(lambda r: r.split(' ')[0])

    calls.loc[calls.call_type.isin(['Phee', 'Trill']), 'call_type'] = 'PheeTrill'
    calls.loc[calls.call_type.isin(['Cheep', 'Chuck']), 'call_type'] = 'CheepChuck'

    sr, _ = wav.read(os.path.join(DATA_LOC, 'ML_Test.wav'))

    X = np.vstack([
        process(*calls.loc[i, ['file', 'start', 'end']])
        for i in calls.index
    ])

    y = np.array(calls.call_type, dtype=str)

    accs = np.zeros(10)
    for seed in trange(len(accs)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        dim_reducer = UMAP(random_state=seed).fit(X_train)
        Z_train = dim_reducer.transform(X_train)
        Z_test = dim_reducer.transform(X_test)

        classifier = KNeighborsClassifier().fit(Z_train, y_train)
        accs[seed] = (classifier.predict(Z_test) == y_test).mean()
    accs *= 100
    print(f'Accuracy:{accs.mean().round(2)}%±{2*accs.std().round(1)}%')
    # 81.15%±4.8%

    Z = np.vstack([Z_train, Z_test])
    plot_y = np.hstack([y_train, y_test])

    plot_df = pd.DataFrame(dict(latent_a=Z[:, 0], latent_b=Z[:, 1], call_type=plot_y))
    sns.scatterplot(data=plot_df, x='latent_a', y='latent_b', hue='call_type')
