
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MODEL_BUNDLE = torchaudio.pipelines.WAV2VEC2_BASE
# W2V = MODEL_BUNDLE.get_model().to(device)
# SR = MODEL_BUNDLE.sample_rate
SR = 44100
softplus = torch.nn.Softplus()

def extract_feats(audio_segments, num_layers=3):
    feats, _ = W2V.extract_features(audio_segments, num_layers=num_layers)

class Files:
    data_loc = '../data_for_expt/'
    lb_data_loc = '../data_for_expt/labelled_data/'

    ml_test = 'ML_Test.wav'
    labels_file = 'Calls_ML_Fix.xlsx'

    hawaii = 'soundscapes/hawaii.wav'
    cali = 'soundscapes/cali_3.wav'
    amazon = 'soundscapes/amazon_3.wav'

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

    def get_samples(self, audio_len=2.5):
        assert audio_len < 5
        segm_len = int(self.nps * audio_len)

        features = []
        labels = []

        l('Processing data.')
        for file in self.features.keys():
            lbs, feats = self.label_ts[file], self.features[file]
            for i in trange(max(0, len(feats)//segm_len)):
                start_idx = i*segm_len # np.random.choice(len(feats) - segm_len - 1)
                end_idx = start_idx + segm_len

                _ft = feats[None, start_idx:end_idx, :].clone()
                _lb = lbs[None, start_idx:end_idx].clone()
                if (len(_ft[0]) == len(_lb[0])) and (len(_lb[0]) == segm_len):
                    features.append(_ft)
                    labels.append(_lb)

        return torch.cat(features, axis=0), torch.cat(labels, axis=0)

    def __getitem__(self, *args):
        """ indexing data[n] is the same as data.get_samples(n) """
        return self.get_samples()

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

    data_loader = AudioDataset(device=device)

    if not os.path.exists('X.pth'):
        X, y = data_loader[...]
        torch.save(X.cpu(), 'X.pth')
        torch.save(y.cpu(), 'y.pth')

    X_full = torch.load('X.pth').to(device)
    y_full = torch.load('y.pth').to(device)

    idx = np.random.choice(
        y_full.mean(axis=1).sort().indices[-350:].cpu().numpy(),
        350, replace=False
    )
    train_idx, test_idx = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]

    X_train = X_full[train_idx, ...]
    y_train = y_full[train_idx, ...]

    X_test = X_full[test_idx, ...]
    y_test = y_full[test_idx, ...].cpu().numpy().reshape(-1)

    classifier = Classifier(data_loader.featurizer.n_mfcc).to(device)

    optimizer = torch.optim.Adam([
        dict(params=classifier.parameters(), lr=0.001),
    ])

    losses = []; iterator = trange(2500, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        y_prob = classifier(X_train)
        loss = -torch.distributions.Bernoulli(y_prob).log_prob(y_train).sum()

        if i % 100 == 0:
            tr_cm = confusion_matrix(y_train.reshape(-1).cpu(), y_prob.round().reshape(-1).detach().cpu(), normalize='all').round(3)*100
            tr_cm = (tr_cm[0, 0] + tr_cm[1, 1]).round(2)

            pred = classifier(X_test).detach().cpu().round().reshape(-1)
            pred[0] = 0; pred[-1] = 1
            cm = confusion_matrix(y_test, pred, normalize='all').round(3)*100
            cm = (cm[0, 0] + cm[1, 1]).round(2)

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm},Te:{cm}')
        loss.backward()
        optimizer.step()

    plt.plot(classifier(data_loader.featurizer(data_loader.audio['ML_Test_3']).T[None, ...])[0].detach().cpu())
    plt.plot(data_loader.label_ts['ML_Test_3'].cpu())

    basic_ml_test_cm = get_confusion_matrix(
        np.array(data_loader.labels.loc[data_loader.labels.file == Files.ml_test.strip('.wav'), ['start', 'end']]),
        simple_classifier(Files.lb_data_loc + Files.ml_test)
    )
