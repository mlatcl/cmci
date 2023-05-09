
import torch
import torch.nn as nn
import torchaudio

import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
np.random.seed(42); torch.manual_seed(42)

import seaborn as sns
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('seaborn-pastel')

from librosa.feature import mfcc
from utils import preprocess_call_labels
from callfinder import CallFinder as CallFinderBasic
from audio.audio_processing import get_spectrum, load_audio_file
from sklearn.metrics import confusion_matrix

from loguru import logger; l = logger.debug

MODEL_BUNDLE = torchaudio.pipelines.WAV2VEC2_XLSR53
SR = MODEL_BUNDLE.sample_rate

class Files:
    data_loc = '../data_for_expt/'
    lb_data_loc = '../data_for_expt/labelled_data/'
    unlb_data_loc = '../data_for_expt/banham_samples/'

    lb_trn_file = 'ML_Test.wav'
    labels_file = 'Calls_ML_Fix.xlsx'

    hawaii = 'soundscapes/hawaii.wav'
    cali = 'soundscapes/cali_3.wav'
    amazon = 'soundscapes/amazon_3.wav'

def load_audio(file_path):
    sr, audio = load_audio_file(file_path)
    audio = torch.tensor(audio)
    audio = torchaudio.functional.resample(audio, sr, SR)
    return audio

def simple_classifier(file_path):
    sr, audio = load_audio_file(file_path)

    S, f, t = get_spectrum(
        start_time=0.0,
        sampling_rate=sr,
        audio=audio,
        segment_length=np.ceil(len(audio)/sr)
    )

    call_finder = CallFinderBasic()
    return call_finder.find_calls(S, f, t)[0]

def get_confusion_matrix(segments_true, segments_pred):
    t_start, t_end, t_step = 0, max(np.max(segments_true), np.max(segments_pred)), 0.05
    t = np.arange(0, t_end//t_step + 1)*t_step + t_start

    arrays = dict(
        true=np.zeros_like(t),
        pred=np.zeros_like(t)
    )
    for name, array in arrays.items():
        segments = segments_true if name == 'true' else segments_pred
        for (t_start, t_end) in segments:
            arrays[name][(t >= t_start) & (t <= t_end)] = 1.0

    return confusion_matrix(arrays['true'], arrays['pred'], normalize='pred').round(3)*100

def preprocessed_segments(file_path):
    sr, audio = load_audio_file(file_path)

    l('segments preprocessor: running the simple classifier.')
    segments = simple_classifier(file_path)

    audio = torch.tensor(audio)
    audio = torchaudio.functional.resample(audio, sr, SR)
    t = np.arange(len(audio))/SR

    l('segments preprocessor: segmenting audio.')
    audio_segments = [
        audio[abs(t - t_start).argmin():abs(t - t_end).argmin()]
        for (t_start, t_end) in tqdm(segments, leave=False)
    ]
    return audio_segments

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self):
        l('Loading labelled data.')
        self._load_labelled_data()

        l('Loading unlabelled data.')
        self._load_unlabelled_data()

        l('Loading soundscapes.')
        self._load_soundscapes()

    def __len__(self):
        return 1

    def _load_soundscapes(self):
        self.soundscapes = dict(
            hawaii=load_audio(Files.data_loc + Files.hawaii)[:(SR * 3600)],
            cali=load_audio(Files.data_loc + Files.cali),
            amazon=load_audio(Files.data_loc + Files.amazon)
        )

    def _load_labelled_data(self):
        labelled_audio = {f.replace('.wav', ''): load_audio(Files.lb_data_loc + f) for f in os.listdir(Files.lb_data_loc) if '.wav' in f}

        calls = pd.read_excel(os.path.join(Files.lb_data_loc, Files.labels_file))
        calls = preprocess_call_labels(calls, keep_only_conures=False)
        self.labels_df = calls.loc[calls.call_type != 'interference', ['file', 'call_type', 'start', 'end']].reset_index(drop=True)

        saved_file_path = Files.lb_data_loc + 'segments_labelled.pth'
        if not os.path.exists(saved_file_path):

            sr, lb_audio = load_audio_file(Files.lb_data_loc + Files.lb_trn_file)

            lb_audio = torch.tensor(lb_audio)
            lb_audio = torchaudio.functional.resample(lb_audio, sr, SR)
            t = np.arange(len(lb_audio))/SR

            segments = np.asarray(self.labels_df.loc[
                self.labels_df.file == Files.lb_trn_file.replace('.wav', ''), ['start', 'end']
            ])

            l('segmenting audio.')
            audio_segments = [
                lb_audio[abs(t - t_start).argmin():abs(t - t_end).argmin()]
                for (t_start, t_end) in tqdm(segments, leave=False)
            ]

            torch.save(audio_segments, saved_file_path)
        self.labelled_segments = torch.load(saved_file_path)
        self.labelled_segments = [a for a in self.labelled_segments if len(a)/SR > 0.05]

    def _load_unlabelled_data(self):
        saved_file_path = Files.unlb_data_loc + 'segments.pth'
        if not os.path.exists(saved_file_path):
            files = [Files.unlb_data_loc + f for f in os.listdir(Files.unlb_data_loc) if '.wav' in f]
            audio_segments = preprocessed_segments(files.pop())
            for another_file in files:
                audio_segments.extend(preprocessed_segments(another_file))
            torch.save(audio_segments, saved_file_path)
        self.unlabelled_segments = torch.load(saved_file_path)

    def get_samples(self, n_samples):
        idx_pos = np.random.choice(len(self.labelled_segments), n_samples)
        idx_mix = np.random.choice(len(self.unlabelled_segments), n_samples)

        X_pos = [self.labelled_segments[i] for i in idx_pos]
        X_pos = [x/x.std() for x in X_pos]
        max_len = max(int(0.5 * SR), max([len(x) for x in X_pos]))

        X_mix = [self.unlabelled_segments[i] for i in idx_mix]
        X_mix = [x/x.std() for x in X_mix]

        pos_samples = self.get_soundscape_samples(n_samples, max_len)
        neg_samples = self.get_soundscape_samples(n_samples, max_len)
        mix_samples = self.get_soundscape_samples(n_samples, max_len)

        for i, x in enumerate(X_pos):
            if len(x) >= max_len:
                mix_samples[i, :] = (mix_samples[i, :] + x[:max_len])/2
            else:
                start_idx = np.random.choice(max_len - len(x))
                end_idx = start_idx + len(x)
                pos_samples[i, start_idx:end_idx] = (pos_samples[i, start_idx:end_idx] + x)/2

        for i, x in enumerate(X_mix):
            if len(x) >= max_len:
                mix_samples[i, :] = (mix_samples[i, :] + x[:max_len])/2
            else:
                start_idx = np.random.choice(max_len - len(x))
                end_idx = start_idx + len(x)
                mix_samples[i, start_idx:end_idx] = (mix_samples[i, start_idx:end_idx] + x)/2

        return (torch.cat([neg_samples, mix_samples, pos_samples], axis=0),
                torch.repeat_interleave(torch.arange(3), n_samples))

    def _get_one_soundscape_sample(self, max_len):
        audio = tuple(self.soundscapes.values())[np.random.choice(len(self.soundscapes))]
        start_idx = np.random.choice(len(audio) - max_len)
        end_idx = start_idx + max_len
        audio_segment = audio[start_idx:end_idx]
        return audio_segment/audio_segment.std()

    def get_soundscape_samples(self, n_samples, max_len):
        return torch.cat([
            self._get_one_soundscape_sample(max_len)[None, :] for _ in range(n_samples)
        ], axis=0)

    def __getitem__(self, n_samples):
        """ indexing data[n] is the same as data.get_samples(n) """
        return get_samples(n_samples)

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_model = MODEL_BUNDLE.get_model()
        self.lstm = nn.LSTM(input_size=n_timepoints, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        torch.cat(self.audio_model.extract_features(x, num_layers=2)[0], axis=2)
        pass

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = AudioDataset()

    basic_ml_test_cm = get_confusion_matrix(
        np.array(data.labels_df.loc[data.labels_df.file == Files.lb_trn_file.replace('.wav', ''), ['start', 'end']]),
        simple_classifier(Files.lb_data_loc + Files.lb_trn_file)
    )
