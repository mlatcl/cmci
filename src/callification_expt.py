
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

DATA_LOC = '../data/Calls For ML/'

def process(f, start, end, n_fft_prop=1/3):
    a = read_audio(f + '.wav')[int(start * sr):int(end * sr)]
    # S = spectrogram(a, nperseg=len(a)//3, noverlap=len(a)//12, fs=sr)[-1]
    S = np.abs(stft(a,
        n_fft=int(len(a) * n_fft_prop),
        hop_length=int(len(a) * n_fft_prop/2
    )))
    print(S.shape)
    mel_features = mfcc(S=S, n_mfcc=20)
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)

    features = np.hstack([
        mel_features.reshape(-1),
        additional_features(S, start, end)
    ])
    return features

def additional_features(spectrogram, start, end):
    duration = end - start
    additional_features = np.hstack([
        duration,
    ])
    return additional_features

@lru_cache(maxsize=10)
def read_audio(f):
    return wav.read(os.path.join(DATA_LOC, f))[1].mean(axis=1)

if __name__ == '__main__':

    calls = pd.read_excel(os.path.join(DATA_LOC, 'Calls_ML_Fix.xlsx'))
    calls.columns = [c.lower().replace(' ', '_') for c in calls.columns]
    calls = calls.loc[~calls.call_type.isna() | (calls.interference == 'Conure')].reset_index(drop=True)
    calls.loc[calls.call_type.isna(), 'call_type'] = 'interference'
    calls = calls.loc[calls.start < calls.end].reset_index(drop=True)
    calls['call_type'] = calls.call_type.apply(lambda r: r.split(' ')[0])

    # calls.loc[calls.call_type.isin(['Phee', 'Trill', 'Whistle']), 'call_type'] = 'LongCalls'
    # calls.loc[calls.call_type.isin(['Cheep', 'Chuck', 'Tsit']), 'call_type'] = 'ShortCalls'

    sr, _ = wav.read(os.path.join(DATA_LOC, 'ML_Test.wav'))

    X = np.vstack([
        process(*calls.loc[i, ['file', 'start', 'end']])
        for i in calls.index
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    y = np.array(calls.call_type, dtype=str)
    le = LabelEncoder()
    le.fit(y)
    y_transformed = le.transform(y)

    accs = np.zeros(10)
    for seed in trange(len(accs)):
        X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=seed)

        dim_reducer = UMAP(random_state=seed).fit(X_train)
        Z_train = dim_reducer.transform(X_train)
        Z_test = dim_reducer.transform(X_test)

        classifier = KNeighborsClassifier().fit(Z_train, y_train)
        accs[seed] = (classifier.predict(Z_test) == y_test).mean()
    accs *= 100
    print(f'Accuracy:{accs.mean().round(2)}%±{2*accs.std().round(1)}%')
    # 84.1% ± 2.8% grouped

    ConfusionMatrixDisplay(
        confusion_matrix(
            le.inverse_transform(y_test),
            le.inverse_transform(classifier.predict(Z_test)),
            normalize='true'
        ).round(2),
        display_labels=le.classes_
    ).plot()

    Z = np.vstack([Z_train, Z_test])
    plot_y = le.inverse_transform(np.hstack([y_train, y_test]))

    plot_df = pd.DataFrame(dict(latent_a=Z[:, 0], latent_b=Z[:, 1], call_type=plot_y))
    sns.scatterplot(data=plot_df, x='latent_a', y='latent_b', hue='call_type', palette='Paired')

    # -------------------------------------------------------------
    # this generates the plot that connnects a saved segment to the classifier

    dim_reducer = UMAP(random_state=42).fit(X)
    Z = dim_reducer.transform(X)

    plot_df = pd.DataFrame(dict(latent_a=Z[:, 0], latent_b=Z[:, 1], call_type=y))
    sns.scatterplot(data=plot_df, x='latent_a', y='latent_b', hue='call_type', palette='Paired')

    import json

    sr, _ = wav.read(os.path.join(DATA_LOC, 'ML_Test.wav'))
    with open('../data/Calls for ML/segments_ML_Test.json', 'rb') as file:
        segments = json.load(file)

    X_unlab = np.vstack([
        process('ML_Test', start, end)
        for (start, end) in segments['data/ML_Test.wav']['segments']
    ])
    X_unlab = (X_unlab - X_unlab.mean(axis=0)) / (X_unlab.std(axis=0) + 1e-12)

    Z_unlab = dim_reducer.transform(X_unlab)

    plot_df = pd.DataFrame(dict(
        latent_a=np.hstack([Z[:, 0], Z_unlab[:, 0]]),
        latent_b=np.hstack([Z[:, 1], Z_unlab[:, 1]]),
        call_type=np.hstack([y, ['unlabelled']*len(X_unlab)])
    ))
    sns.scatterplot(data=plot_df, x='latent_a', y='latent_b', hue='call_type', palette='Paired', alpha=0.5)
