
import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange
import matplotlib.pyplot as plt; plt.ion()
from librosa.feature import mfcc

from audio.audio_processing import get_spectrum, load_audio_file
from callfinder import CallFinder as CallFinderBasic
from utils import preprocess_call_labels

DATA_LOC = '../data/calls_for_ml/'

def spectral_flatness(S, eps=1e-10):
    spec = np.exp(S) - eps
    num = np.exp(np.log(spec + eps).mean(axis=0)) - eps
    denom = spec.mean(axis=0)
    return (np.log(num + eps) - np.log(denom + eps))

if __name__ == '__main__':
    CALLS_FILE='Calls_ML_Fix.xlsx'
    AUDIO_FILE='ML_Test.wav'

    calls = pd.read_excel(os.path.join(DATA_LOC, CALLS_FILE))
    calls = preprocess_call_labels(calls, keep_only_conures=True)
    calls = calls.loc[(calls.file == 'ML_Test')
         # & (calls.call_type != 'interference')
    ].reset_index(drop=True)

    sampling_rate, audio = load_audio_file(DATA_LOC + AUDIO_FILE)

    S, f, t = get_spectrum(
        start_time=0.0,
        sampling_rate=sampling_rate,
        audio=audio,
        segment_length=np.ceil(len(audio)/sampling_rate)
    )

    call_finder = CallFinderBasic()
    segments, thresholded_spectrum, _, _ = call_finder.find_calls(S, f, t)

    y_pred = np.zeros_like(t)
    for (t_start, t_end) in segments:
        y_pred[(t >= t_start) & (t <= t_end)] = 1.0

    y_true = np.zeros_like(t)
    for i in range(len(calls)):
        t_start, t_end = calls.loc[i, ['start', 'end']]
        y_true[(t >= t_start) & (t <= t_end)] = 1.0

    flatness = spectral_flatness(S)
    mel_coefs = mfcc(S=S, sr=sampling_rate)[3:, :]

    binarised_spec_feat = thresholded_spectrum[(f >= call_finder.band_to_consider[0]) & (f <= call_finder.band_to_consider[1])]

    n_reduced_feats = 20
    reducer = np.kron(np.eye(n_reduced_feats), np.ones((len(binarised_spec_feat)//n_reduced_feats, 1)))
    reducer = np.concatenate([reducer,
        np.repeat(reducer[[-1], :], len(binarised_spec_feat) - len(reducer), axis=0)
    ], axis=0)

    X_feats = np.vstack([
        reducer.T @ binarised_spec_feat,
        mel_coefs,
        # flatness
    ]).T

    n_lag = 3
    X_feats_lag = np.concatenate([
        np.zeros((n_lag, len(X_feats.T))),
        X_feats[:-n_lag, :]
    ], axis=0)

    X_feats = np.concatenate([X_feats, X_feats_lag], axis=1)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=int(1e6), verbose=1).fit(X_feats, y_true)
    y_pred_reg = model.predict_proba(X_feats)[:, 1]

    plt.plot(y_pred[:1000])
    plt.plot((y_pred_reg[:1000] > 0.8) + 1)
    plt.plot(y_true[:1000] + 2)