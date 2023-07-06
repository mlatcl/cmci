
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

MODEL_BUNDLE = torchaudio.pipelines.WAV2VEC2_BASE
SR = MODEL_BUNDLE.sample_rate
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
    def __init__(self, n_transforms=12, device='cpu'):
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

        l("Loading wave2vec2.")
        self.featurizer = MODEL_BUNDLE.get_model().eval().to(device)
        self.n_transforms = n_transforms

        l("Preprocessing label time series.")
        self.nps = self.featurizer.extract_features(torch.zeros(1, SR).to(device), num_layers=1)[0][0].shape[1] + 1

        self.label_ts = {k: None for k in self.audio.keys()}
        ts = {k: int(idx * self.nps / SR) for k, (idx, _) in self.audio_lens.items()}
        for k, t in ts.items():
            ts[k] = torch.arange(t).to(device) * self.audio_lens[k][-1] / t

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
            len_idx, len_t = self.audio_lens[file]

            start_idx = min(max(0, np.random.choice(torch.where(self.label_ts[file])[0].cpu()) - int(SR*0.05)), len_idx - int(audio_len*SR) - 1)
            end_idx = start_idx + int(audio_len*SR)

            features.append(self.featurizer.extract_features(
                self.audio[file][None, start_idx:end_idx],
                num_layers=self.n_transforms
            )[0][-1].detach())

            start_t = int(audio_len * self.nps * start_idx/len_idx)
            end_t = start_t + int(self.nps*audio_len) - 1

            labels.append(self.label_ts[file][None, start_t:end_t].clone())

        return torch.cat(features, axis=0), torch.cat(labels, axis=0)

    def __getitem__(self, n_samples):
        """ indexing data[n] is the same as data.get_samples(n) """
        return self.get_samples(n_samples)

class Classifier(torch.nn.Module):
    def __init__(self, num_lstm=3):
        super().__init__()
        self.num_lstm = num_lstm
        self.lstm = torch.nn.LSTM(MODEL_BUNDLE._params['encoder_embed_dim'], 10, num_lstm, batch_first=True, )
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_lstm, x.size(0), 10).to(x.device)
        c0 = torch.zeros(self.num_lstm, x.size(0), 10).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out).sigmoid()[..., 0]
        return out

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_loader = AudioDataset(n_transforms=2, device=device)

    if not os.path.exists('X.pth'):
        X, y = data_loader[10000]
        torch.save(X.cpu(), 'X.pth')
        torch.save(y.cpu(), 'y.pth')

    X = torch.load('X.pth').to(device)
    y = torch.load('y.pth').to(device)

    classifier = Classifier().to(device)

    test_feats = data_loader.featurizer.extract_features(
        data_loader.audio['ML_Test_3'][None, ...],
        num_layers=data_loader.n_transforms
    )[0][-1].detach()
    y_test = data_loader.label_ts['ML_Test_3'].cpu().numpy()

    optimizer = torch.optim.Adam([
        dict(params=classifier.parameters(), lr=0.01),
    ])

    losses = []; iterator = trange(2000, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        idx = np.random.choice(len(X), 500)
        X, y = X[idx], y[idx]

        y_prob = classifier(X)
        loss = -torch.distributions.Bernoulli(y_prob).log_prob(y).sum()

        tr_cm = confusion_matrix(y.reshape(-1).cpu(), y_prob.round().reshape(-1).detach().cpu(), normalize='all').round(3)*100

        if i % 100 == 0:
            pred = classifier(test_feats)[0].detach().cpu().round()
            pred[0] = 1
            cm = confusion_matrix(y_test, pred, normalize='all').round(3)*100

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm.shape},Te:{cm[0, 0]+cm[1, 1]}')
        loss.backward()
        optimizer.step()

    basic_ml_test_cm = get_confusion_matrix(
        np.array(data_loader.labels.loc[data_loader.labels.file == Files.lb_trn_file.strip('.wav'), ['start', 'end']]),
        simple_classifier(Files.lb_data_loc + Files.lb_trn_file)
    )
