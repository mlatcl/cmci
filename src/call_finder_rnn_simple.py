
import torch
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

SR = 44100
softplus = torch.nn.Softplus()

class Files:
    data_loc = '../data_for_expt/'
    lb_data_loc = '../data_for_expt/labelled_data/'

    lb_trn_file = 'ML_Test.wav'
    labels_file = 'Calls_ML_Fix.xlsx'

def load_audio(file_path):
    sr, audio = load_audio_file(file_path)
    audio = torchaudio.functional.resample(torch.tensor(audio), sr, SR)
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

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.audio = {
            f.replace('.wav', ''): load_audio(Files.lb_data_loc + f).to(device) for f in os.listdir(Files.lb_data_loc) if '.wav' in f
        }
        self.audio_lens = {k: (len(a), len(a)/SR) for k, a in self.audio.items()}

        l("Processing labelled data.")
        calls = pd.read_excel(os.path.join(Files.lb_data_loc, Files.labels_file))
        calls = preprocess_call_labels(calls, keep_only_conures=False)
        self.labels = calls.loc[
            (calls.call_type != 'interference'),
            ['file', 'call_type', 'start', 'end']
        ].reset_index(drop=True)

        l("Computing mfcc.")
        self.featurizer = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=40).to(device)

        l("Preprocessing label time series.")
        self.nps = self.featurizer(torch.zeros(1, SR).to(device)).shape[-1]

        self.features = {k: self.featurizer(a).T for k, a in self.audio.items()}

        self.label_ts = {k: None for k in self.audio.keys()}
        ts = {k: self.audio_lens[k][-1]*torch.arange(f.shape[0]).to(device)/f.shape[0] for k, f in self.features.items()}
        for k in ts.keys():
            temp_df = np.asarray(self.labels.loc[self.labels.file == k, ['start', 'end']])
            self.label_ts[k] = torch.zeros_like(ts[k])
            for start, end in temp_df:
                self.label_ts[k][(ts[k] >= start) & (ts[k] < end)] = 1.0

    def __len__(self):
        return 1

    def get_samples(self, n_samples, audio_len=2.5):
        assert audio_len < 5
        files = []; probs = np.empty(len(self.audio), dtype=float)
        for i, (f, ln) in enumerate(self.audio_lens.items()):
            files.append(f)
            probs[i] = ln[0] if f != 'ML_Test_3' else 0.0
        probs /= sum(probs)

        files_choose = np.random.choice(files, n_samples, replace=True, p=probs)

        features = []
        labels = []

        l('Processing data.')
        for i, file in tqdm(enumerate(files_choose)):
            segm_len = int(audio_len*SR/self.nps)
            lbs, feats = self.label_ts[file], self.features[file]

            start_idx = np.random.choice(len(feats) - segm_len - 1)
            end_idx = start_idx + segm_len

            features.append(feats[None, start_idx:end_idx, :].clone())
            labels.append(lbs[None, start_idx:end_idx].clone())

        return torch.cat(features, axis=0), torch.cat(labels, axis=0)

    def __getitem__(self, n_samples):
        """ indexing data[n] is the same as data.get_samples(n) """
        return self.get_samples(n_samples)

class Classifier(torch.nn.Module):
    def __init__(self, num_inp, num_lstm=3):
        super().__init__()
        self.num_lstm = num_lstm
        self.lstm = torch.nn.LSTM(num_inp, 10, num_lstm, batch_first=True)
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_lstm, x.size(0), 10).to(x.device)
        c0 = torch.zeros(self.num_lstm, x.size(0), 10).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out).sigmoid()[..., 0]
        return out

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader = AudioDataset(device=device)

    if not os.path.exists('X.pth'):
        X, y = data_loader[50000]
        torch.save(X.cpu(), 'X.pth')
        torch.save(y.cpu(), 'y.pth')

    X_full = torch.load('X.pth').to(device)
    y_full = torch.load('y.pth').to(device)

    classifier = Classifier(data_loader.featurizer.n_mfcc).to(device)

    test_feats = data_loader.features['ML_Test_3'][None, ...]
    y_test = data_loader.label_ts['ML_Test_3'].cpu().numpy()

    optimizer = torch.optim.Adam([
        dict(params=classifier.parameters(), lr=0.01),
    ])

    losses = []; iterator = trange(2000, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        # idx = np.random.choice(len(X_full), 500)
        X, y = X_full[:5000], y_full[:5000]

        y_prob = classifier(X)
        loss = -torch.distributions.Bernoulli(y_prob).log_prob(y).sum()

        if i % 100 == 0:
            tr_cm = confusion_matrix(y.reshape(-1).cpu(), y_prob.round().reshape(-1).detach().cpu(), normalize='all').round(3)*100

            pred = classifier(test_feats)[0].detach().cpu().round()
            pred[0] = 1
            cm = confusion_matrix(y_test, pred, normalize='all').round(3)*100

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm[0, 0] + tr_cm[1, 1]},Te:{cm[0, 0]+cm[1, 1]}')
        loss.backward()
        optimizer.step()

    basic_ml_test_cm = get_confusion_matrix(
        np.array(data_loader.labels.loc[data_loader.labels.file == Files.lb_trn_file.strip('.wav'), ['start', 'end']]),
        simple_classifier(Files.lb_data_loc + Files.lb_trn_file)
    )
