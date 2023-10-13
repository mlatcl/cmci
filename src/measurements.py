
import os, warnings
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()

from scipy.interpolate import splrep, BSpline

from scipy.signal import stft
from audio.audio_processing import get_spectrum, load_audio_file
from call_finder_rnn_simple import AudioDataset, device, Files

from librosa import pyin, note_to_hz

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('seaborn-v0_8-pastel')

def process_row(x, min_freq=note_to_hz('C3')):    
    sr, audio = load_audio_file(
        os.path.join(Files.lb_data_loc, x.file + '.wav')
    )

    audio = audio[int(x.start*sr):int(x.end*sr)]

    f, t, S = stft(audio, nperseg=sr//20, fs=sr)
    S = np.log(np.abs(S) + 1e-10)

    S = S[f >= min_freq, :]
    f = f[f >= min_freq]

    S_smooth = np.concatenate([BSpline(*splrep(f, S[:, i], s=len(f)))(f)[:, None] for i in range(len(S.T))], axis=1)

    S = S - S_smooth

    feats = CallFeatures(S, f, t)
    f0 = np.nanmedian(pyin(audio, sr=sr, fmin=min_freq, fmax=note_to_hz('C9'))[0])

    bar.update()
    return pd.Series(dict(
        call=x.call_type,
        duration=feats.duration(),
        min_freq=feats.min_freq(),
        max_freq=feats.max_freq(),
        f_max=feats.freq_of_max_power(),
        f0=f0
    ))

class CallFeatures:
    def __init__(self, S, f, t):
        self.S = S
        self.f = f
        self.t = t

    def _freq_at_time(self, position):
        return self.f[self.S[:, int(np.round((self.S.shape[1]-1)*position))].argmax()]

    def min_freq(self):
        return self._freq_at_time(0.15)

    def max_freq(self):
        return self._freq_at_time(0.85)

    def freq_of_max_power(self):
        return self.f[np.unravel_index(self.S.argmax(), self.S.shape)[0]]

    def duration(self):
        return self.t.max() - self.t.min()

if __name__ == '__main__':

    dataset = AudioDataset(device=device)
    calls = dataset.labels.copy()
    calls.loc[calls.call_type == 'Jagged Trill', 'call_type'] = 'Jagged'
    calls.loc[calls.call_type == 'Resonating Note', 'call_type'] = 'Resonate'

    # processed = calls.apply(process_row, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bar = trange(len(calls))
        processed = calls.loc[:300].apply(process_row, axis=1)

    # i = np.random.choice(len(calls))
    # x = calls.loc[100]

    processed.groupby('call').mean()

