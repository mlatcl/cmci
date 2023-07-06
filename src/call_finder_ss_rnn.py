
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
from call_finder_rnn_simple import CallFinder as CallFinderRNN

from loguru import logger; l = logger.debug

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SR = 44100
softplus = torch.nn.Softplus()

class Files:
    data_loc = '../data_for_expt/'
    lb_data_loc = '../data_for_expt/labelled_data/'
    unlb_data_loc = '../data_for_expt/banham_samples/'

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
        self._init_labelled_data(device)
        self._init_unlabelled_data(device)

    def _init_labelled_data(self, device):
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

    def _init_unlabelled_data(self, device):
        self.unlb_audio = {
            f.replace('.wav', ''): load_audio(Files.unlb_data_loc + f).to(device) for f in os.listdir(Files.unlb_data_loc) if '.wav' in f
        }
        self.unlb_audio_lens = {k: (len(a), len(a)/SR) for k, a in self.unlb_audio.items()}

        l("Processing unlabelled data.")
        if not os.path.exists(Files.unlb_data_loc + 'labels.csv'):
            calls = {
                f.replace('.wav', ''): simple_classifier(Files.unlb_data_loc + f) for f in tqdm(os.listdir(Files.unlb_data_loc)) if '.wav' in f
            }
            def df_maker(key, call_array):
                calls_df = pd.DataFrame(call_array)
                calls_df.columns = ['start', 'end']
                calls_df['file'] = key
                return calls_df

            self.unlb_labels = pd.concat([df_maker(key, call_array) for key, call_array in calls.items()]).reset_index(drop=True)
            self.unlb_labels.to_csv(Files.unlb_data_loc + 'labels.csv', index=False)
        else:
            self.unlb_labels = pd.read_csv(Files.unlb_data_loc + 'labels.csv')

        self.unlb_features = {k: self.featurizer(a).T for k, a in self.unlb_audio.items()}

        self.unlb_label_ts = {k: None for k in self.unlb_audio.keys()}
        ts = {k: self.unlb_audio_lens[k][-1]*torch.arange(f.shape[0]).to(device)/f.shape[0] for k, f in self.unlb_features.items()}
        for k in ts.keys():
            temp_df = np.asarray(self.unlb_labels.loc[self.unlb_labels.file == k, ['start', 'end']])
            self.unlb_label_ts[k] = torch.zeros_like(ts[k])
            for start, end in temp_df:
                self.unlb_label_ts[k][(ts[k] >= start) & (ts[k] < end)] = 1.0

    def __len__(self):
        return 1

    def get_samples(self, audio_len=2.5):
        assert audio_len < 5
        segm_len = int(self.nps * audio_len)

        features = []
        labels = []

        l('Processing data.')
        files_to_process = [f for f in self.features.keys() if f != 'ML_Test_3']

        for file in files_to_process:
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

    def get_unlabelled_samples(self, audio_len=2.5):
        assert audio_len < 5
        segm_len = int(self.nps * audio_len)

        features = []
        labels = []

        l('Processing data.')
        for file in self.unlb_features.keys():
            lbs, feats = self.unlb_label_ts[file], self.unlb_features[file]
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
        X, y = data_loader.get_samples()
        X_unlb, y_unlb = data_loader.get_unlabelled_samples()

        torch.save(X.cpu(), 'X.pth')
        torch.save(X_unlb.cpu(), 'X_unlb.pth')
        torch.save(y.cpu(), 'y.pth')
        torch.save(y_unlb.cpu(), 'y_unlb.pth')

    X_full = torch.load('X.pth').to(device)
    X_unlb = torch.load('X_unlb.pth').to(device)
    y_full = torch.load('y.pth').to(device)
    y_unlb = torch.load('y_unlb.pth').to(device)

    idx = np.random.choice(
        y_full.mean(axis=1).sort().indices[-400:].cpu().numpy(),
        400, replace=False
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

    losses = []; iterator = trange(1000, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        unlb_idx = np.random.choice(len(X_unlb), 500)
        _X = torch.cat([X_train, X_unlb[unlb_idx]], axis=0)
        _y = torch.cat([y_train, y_unlb[unlb_idx]], axis=0)

        y_prob = classifier(_X)
        loss = -torch.distributions.Bernoulli(y_prob).log_prob(_y).sum()

        if i % 100 == 0:
            tr_cm = confusion_matrix(y_train.reshape(-1).cpu(), y_prob[:len(y_train)].round().reshape(-1).detach().cpu(), normalize='all').round(3)*100
            tr_cm = (tr_cm[0, 0] + tr_cm[1, 1]).round(2)

            pred = classifier(X_test).detach().cpu().round().reshape(-1)
            pred[0] = 0; pred[-1] = 1
            cm = confusion_matrix(y_test, pred, normalize='all').round(3)*100
            cm = (cm[0, 0] + cm[1, 1]).round(2)

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm},Te:{cm}')
        loss.backward()
        optimizer.step()

    n_semi_sup_train_loops = 5
    for _ in range(n_semi_sup_train_loops):
        y_unlb = classifier(X_unlb).detach().round()

        losses = []; iterator = trange(250, leave=False)
        for i in iterator:
            optimizer.zero_grad()

            unlb_idx = np.random.choice(len(X_unlb), 500)
            _X = torch.cat([X_train, X_unlb[unlb_idx]], axis=0)
            _y = torch.cat([y_train, y_unlb[unlb_idx]], axis=0)

            y_prob = classifier(_X)
            loss = -torch.distributions.Bernoulli(y_prob).log_prob(_y).sum()

            if i % 100 == 0:
                tr_cm = confusion_matrix(y_train.reshape(-1).cpu(), y_prob[:len(y_train)].round().reshape(-1).detach().cpu(), normalize='all').round(3)*100
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

    plt.plot(data_loader.unlb_label_ts['081017-010'][:2000].cpu())
    plt.plot(classifier(data_loader.unlb_features['081017-010'][None, ...])[0].detach().cpu().numpy()[:2000])

    basic_ml_test_cm = get_confusion_matrix(
        np.array(data_loader.labels.loc[data_loader.labels.file == Files.ml_test.strip('.wav'), ['start', 'end']]),
        simple_classifier(Files.lb_data_loc + Files.ml_test)
    )
