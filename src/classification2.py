import os
import numpy as np
import pandas as pd
from tqdm import trange
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from librosa.feature import mfcc
from librosa.core.spectrum import stft

from scipy.io import wavfile as wav
from scipy.signal import spectrogram

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader



SR = 44100
DATA_LOC = '../data/Calls for ML/labelled_data/'


class Files:
    data_loc = '../data/Calls for ML/'

    # create symlinks so that all the data can be seen from labelled_data
    lb_data_loc = '../data/Calls for ML/labelled_data/'

    state_dict = '../data/Calls for ML/simple_rnn_sd.pth'

    ml_test = 'ML_Test.wav'
    labels_file = 'Calls_ML.xlsx'


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.audio = {
            f.replace('.wav', ''): self.load_audio(Files.lb_data_loc + f).to(device) for f in os.listdir(Files.lb_data_loc) if '.wav' in f
        }
        self.audio_lens = {k: (len(a), len(a)/SR) for k, a in self.audio.items()}

        calls = pd.read_excel(os.path.join(Files.lb_data_loc, Files.labels_file))
        calls = calls.loc[(calls.Call_Type != 'interference'), ['File', 'Call_Type', 'Start', 'End']]
        calls = calls.loc[~calls.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
        # calls['File'] = 'Calls_ML'
        calls.columns = calls.columns.str.lower()

        calls_shaldon = pd.read_excel(os.path.join(Files.lb_data_loc, 'Shaldon_Training_Labels.xlsx'))
        calls_shaldon = calls_shaldon.loc[~calls_shaldon.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
        calls_shaldon['File'] = 'Shaldon_Combined'
        calls_shaldon.columns = calls_shaldon.columns.str.lower()

        calls_blackpool = pd.read_excel(os.path.join(Files.lb_data_loc, 'Blackpool_Labels.xlsx'))
        calls_blackpool = calls_blackpool.loc[~calls_blackpool.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
        calls_blackpool['File'] = 'Blackpool_Combined_FINAL'
        calls_blackpool.columns = calls_blackpool.columns.str.lower()

        labels = pd.concat([calls, calls_shaldon, calls_blackpool], axis=0).reset_index(drop=True)

        labels.loc[labels.call_type.isin(['Phee', 'Trill', 'Whistle']), 'call_type'] = 'LongCalls'
        labels.loc[labels.call_type.isin(['Cheep', 'Chuck', 'Tsit']), 'call_type'] = 'ShortCalls'

        # Remove calls that have length 0
        self.labels = labels.loc[labels.end - labels.start > 0].reset_index(drop=True)

        self.X = np.vstack([
            self.process_file(*self.labels.loc[i, ['file', 'start', 'end']], sr=SR)
            for i in self.labels.index
        ])

        self.y = np.array(self.labels.call_type, dtype=str)
        self.le = LabelEncoder()
        self.le.fit(self.y)
        self.y_transformed = self.le.transform(self.y)


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, idx):
        return self.X[idx], self.y_transformed[idx]
    
    
    def load_audio(self, file_path):
        sr, audio = self.load_audio_file(file_path)
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, SR)
        return audio
    

    def process_file(self, f, start, end, sr, n_fft_prop=1/3):
        a = self.load_audio(os.path.join(DATA_LOC, f + '.wav'))[int(start * sr):int(end * sr)].numpy()
        S = np.abs(stft(a,
            n_fft=int(len(a) * n_fft_prop),
            hop_length=int(len(a) * n_fft_prop/2
        )))
        mel_features = mfcc(S=S, n_mfcc=20)
        mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)

        features = np.hstack([
            mel_features.reshape(-1),
            self.additional_features(start, end)
        ])
        return features


    def additional_features(self, start, end):
        duration = end - start
        additional_features = np.hstack([
            duration,
        ])
        return additional_features

    
    @staticmethod
    @lru_cache(maxsize=100)
    def load_audio_file(filename):
        sr, audio = wav.read(filename)
        if len(audio.shape) == 2:
            audio = audio[:, 0]  # take the first channel
        audio = audio.astype('f')/1000  # scale values down by 1000.
        return sr, audio
    

class TorchStandardScaler():
    def fit(self, data):
    # check if data.dataset.X is a numpy array and if so convert to torch tensor
        if type(data.dataset.X) == np.ndarray:
            data.dataset.X = torch.from_numpy(data.dataset.X)
        self.mean = data.dataset.X.mean(dim=0, keepdim=True)
        self.std = data.dataset.X.std(dim=0, keepdim=True)


    def transform(self, data):
        # check if data.dataset.X is a numpy array and if so convert to torch tensor
        if type(data.dataset.X) == np.ndarray:
            data.dataset.X = torch.from_numpy(data.dataset.X)
        data.dataset.X = (data.dataset.X - self.mean) / self.std

        return data
    

class Classifier(torch.nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.Softmax(dim=1)(x)
        return x
    

if __name__ == "__main__":

    TRAIN_PROPORTION = 0.8
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100

    
    # check to see if dataset exists in the parent directory
    if os.path.exists('../dataset.pt'):
        print("Loading dataset...\n")
        dataset = torch.load('../dataset.pt')
    else:
        print("Creating dataset...\n")
        dataset = AudioDataset(DEVICE)

        # Save dataset
        torch.save(dataset, '../dataset.pt')

    train_size = int(TRAIN_PROPORTION * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1337))

    scaler = TorchStandardScaler()
    scaler.fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)

    print(dataset.labels.call_type.value_counts(), '\n')

    print(
        f'Train size: {len(train_dataset)}\n'
        f'Test size: {len(test_dataset)}\n'
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Classifier(num_classes=len(dataset.le.classes_), input_size=dataset.X.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X.float())
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: {loss.item():.6f}')

    print("\nTesting model...")
    model.eval()
    
    y_pred = []
    y_true = []
    for X_batch, y_batch in test_dataloader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        y_pred.append(model(X_batch.float()).argmax(dim=1).detach().numpy())
        y_true.append(y_batch.detach().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    print(f'Accuracy: {100*accuracy_score(y_true, y_pred):.2f}%\n')

    # print unique classes and counts from dataset.labels
    
    
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay(
            confusion_matrix(
                dataset.le.inverse_transform(y_true),
                dataset.le.inverse_transform(y_pred),
                normalize='true'
            ).round(2),
            display_labels=dataset.le.classes_
        ).plot(xticks_rotation=90, values_format='.2f', cmap='Blues', ax=ax)
    
    plt.savefig('../confusion_matrix.png', dpi=300, bbox_inches='tight')

    