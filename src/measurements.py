
import os
import numpy as np
import pandas as pd
import seaborn as sns
import json
from tqdm import trange
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from umap import UMAP
from librosa.feature import mfcc
from librosa.core.spectrum import stft
from scipy.io import wavfile as wav
from scipy.signal import spectrogram, periodogram
import librosa
import torch, torchaudio

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from utils import preprocess_call_labels
from audio.audio_processing import get_spectrum

from scipy.interpolate import splrep, BSpline
from scipy.signal import find_peaks

DATA_LOC = '../data_for_expt/labelled_data/'

def process_file_old(f, start, end, sr, n_fft_prop=1/3):
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

def process_file(f, start, end, sr, n_fft_prop=1/3):
    a = read_audio(f + '.wav')[int(start * sr):int(end * sr)]
    return a

def additional_features(spectrogram, start, end):
    duration = end - start
    additional_features = np.hstack([
        duration,
    ])
    return additional_features

@lru_cache(maxsize=50)
def read_audio(f):
    return wav.read(os.path.join(DATA_LOC, f))[1].mean(axis=1)

if __name__ == '__main__':
    CALLS_FILE='Calls_ML_Fix.xlsx'
    AUDIO_FILE='ML_Test.wav'

    sr, _ = wav.read(os.path.join(DATA_LOC, AUDIO_FILE))

    calls = pd.read_excel(os.path.join(DATA_LOC, CALLS_FILE))
    calls = preprocess_call_labels(calls)

    calls = calls.loc[calls.call_type != 'interference'].reset_index(drop=True)

    audio_a = process_file(*calls.loc[0, ['file', 'start', 'end']], sr=sr)

    f0 = librosa.yin(audio_a, fmin=400, fmax=7800, sr=sr, frame_length=sr//20).max()

    f, S = periodogram(audio_a, fs=sr)
    S = np.log(np.abs(S) + 1e-10)

    smooth_spec = BSpline(*splrep(f, S, s=3200))(f)
    peaks = find_peaks(smooth_spec)[0]
    f[peaks]
