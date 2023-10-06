
import os, torch, wandb
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from librosa.feature import mfcc
from librosa.core.spectrum import stft
from scipy.io import wavfile as wav

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from call_finder_rnn_simple import AudioDataset, Files, device

softmax = torch.nn.Softmax(dim=1)

def feature(a, n_fft_prop=1/3):
    S = np.abs(stft(a,
        n_fft=int(len(a) * n_fft_prop),
        hop_length=int(len(a) * n_fft_prop/2
    )))

    mel_features = mfcc(S=S, n_mfcc=20)
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)
    return mel_features.reshape(-1)

def process_file(f, start, end, data_loc=Files.lb_data_loc):
    sr, a = read_audio(f, data_loc=data_loc)
    a = a[int(start * sr):int(end * sr)]
    return feature(a)

@lru_cache(maxsize=50)
def read_audio(f, data_loc=Files.lb_data_loc):
    sr, audio = wav.read(os.path.join(data_loc, f))
    if len(audio.shape) == 2:
        return sr, audio.mean(axis=1)
    else:
        return sr, audio.astype(float)

class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.nnet = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.Softplus(),
            torch.nn.Linear(100, 100),
            torch.nn.Softplus(),
            torch.nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.nnet(x)
        return softmax(x)

# class Classifier(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(5 * 1 * 32, 32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, num_classes)
#         )

#     def forward(self, x):
#         x = x.reshape(-1, 1, 20, 7)
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return torch.nn.functional.softmax(x, dim=-1)

class ClassifierPipeline:
    def __init__(self):
        state_dict = torch.load(Files.classifier_state)
        num_classes, input_size = state_dict['nnet.0.weight'].shape
        num_classes = int(num_classes/3)

        self.classifier = Classifier(input_size, num_classes)
        self.classifier.load_state_dict(state_dict)
        self.classifier.to(device)

        self.le = LabelEncoder()
        self.le.classes_ = np.load(Files.classifier_labels)

    def _predict_one(self, f, start, stop, data_loc=Files.lb_data_loc):
        X = torch.tensor(
            process_file(f, start, stop, data_loc=data_loc)
        ).to(device).float()[None, ...]

        y = self.le.inverse_transform(
            self.classifier(X).argmax(dim=1).cpu().detach().numpy()
        )
        return y

    def predict(self, f, starts, stops, data_loc=Files.lb_data_loc):
        return np.hstack([ \
            self._predict_one(f, starts[i], stops[i], data_loc=data_loc) \
            for i in range(len(starts)) \
        ])

if __name__ == '__main__':

    data_loader = AudioDataset(device=device)
    calls = data_loader.labels.copy()

    calls = calls.loc[calls.end > calls.start].reset_index(drop=True)
    calls.loc[calls.call_type == 'Resonating Note', 'call_type'] = 'Resonate'

    # Reclassify call clusters
    calls.loc[calls.call_type.isin(['Phee', 'Trill', 'Whistle']), 'call_type'] = 'LongCalls'
    calls.loc[calls.call_type.isin(['Cheep', 'Chuck', 'Tsit']), 'call_type'] = 'ShortCalls'

    calls['file_with_ext'] = calls['file'] + '.wav'
    calls['orig'] = False

    calls_first_half = calls.copy()
    calls_first_half['end'] = (calls_first_half['end'] - calls_first_half['start'])/2 + calls_first_half['start']

    calls_sec_half = calls.copy()
    calls_sec_half['start'] = calls_sec_half['end'] - (calls_sec_half['end'] - calls_sec_half['start'])/2

    shift = 0.2

    calls_shifted = calls.copy()
    calls_shifted['start'] += np.random.uniform(-shift, shift, len(calls))
    calls_shifted['end'] += np.random.uniform(-shift, shift, len(calls))
    calls_shifted = calls_shifted.loc[(calls_shifted.end > calls_shifted.start) & (calls_shifted.start > 0)]

    if not os.path.exists('../data/Calls for ML/labelled_data/blackpool_hi.wav'):
        print('making fake hawaii data')
        from scipy.io import wavfile as wav
        import librosa

        monkeys, sr = librosa.load('../data/Calls for ML/labelled_data/Blackpool_Combined_FINAL.wav', sr=44100)
        soundscape, sr = librosa.load('../data/Calls for ML/unlabelled_data/hawaii.wav', sr=sr)

        monkeys = monkeys*0.25 + 0.75*soundscape[:len(monkeys)]
        wav.write('../data/Calls for ML/labelled_data/blackpool_hi.wav', 44100, monkeys)

        monkeys, sr = librosa.load('../data/Calls for ML/labelled_data/Shaldon_Combined.wav', sr=44100)
        soundscape, sr = librosa.load('../data/Calls for ML/unlabelled_data/hawaii.wav', sr=sr)

        monkeys = monkeys*0.25 + 0.75*soundscape[:len(monkeys)]
        wav.write('../data/Calls for ML/labelled_data/shaldon_hi.wav', 44100, monkeys)

    fake_hawaii = calls.copy()
    fake_hawaii = fake_hawaii.loc[fake_hawaii.file.isin(['Blackpool_Combined_FINAL', 'Shaldon_Combined'])].reset_index(drop=True)
    fake_hawaii.loc[fake_hawaii.file == 'Shaldon_Combined', 'file'] = 'shaldon_hi'
    fake_hawaii.loc[fake_hawaii.file == 'Blackpool_Combined_FINAL', 'file'] = 'blackpool_hi'
    fake_hawaii.loc[fake_hawaii.file == 'Shaldon_Combined', 'file_with_ext'] = 'shaldon_hi.wav'
    fake_hawaii.loc[fake_hawaii.file == 'Blackpool_Combined_FINAL', 'file_with_ext'] = 'blackpool_hi.wav'

    calls['orig'] = True
    calls = pd.concat([calls, calls_first_half, calls_sec_half, calls_shifted, fake_hawaii], axis=0).reset_index(drop=True)

    X = [process_file(*calls.loc[i, ['file_with_ext', 'start', 'end']]) \
         for i in tqdm(calls.index)]

    median_size = np.median([len(x) for x in X])
    print(pd.crosstab(np.array([len(x) for x in X]), 0))
    print(median_size)
    calls['drop'] = [(len(x) != median_size) for x in X]
    calls = calls.loc[~calls['drop']].reset_index(drop=True)

    X = np.vstack([
        process_file(*calls.loc[i, ['file_with_ext', 'start', 'end']]) \
        for i in tqdm(calls.index)
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    y = np.array(calls.call_type, dtype=str)
    le = LabelEncoder()
    le.fit(y)
    y_transformed = le.transform(y)

    np.save(Files.classifier_labels, le.classes_)
    torch.save((X, y_transformed), Files.classification_data)

    orig = np.asarray(calls.orig)
    X_train, X_test, y_train, y_test, _, orig_test = \
        train_test_split(X, y_transformed, orig, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)

    classifier = Classifier(X.shape[1], len(le.classes_)).to(device)
    # classifier = Classifier(len(le.classes_)).to(device)

    optimizer = torch.optim.Adam([
        dict(params=classifier.parameters(), lr=0.005),
    ])

    wandb.init(project="monke")

    losses = []; iterator = trange(20000, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        idx = np.random.choice(len(X_train), 500)

        y_prob = classifier(X_train[idx])
        loss = -torch.distributions.Categorical(y_prob).log_prob(y_train[idx]).sum()

        tr_cm = confusion_matrix(y_train[idx].cpu(), y_prob.argmax(dim=1).detach().cpu(), normalize='all')*100
        tr_cm = tr_cm[range(len(tr_cm)), range(len(tr_cm))].sum().round(2)

        te_cm = confusion_matrix(y_test[orig_test], classifier(X_test).argmax(dim=1).detach().cpu()[orig_test], normalize='all')*100
        te_cm = te_cm[range(len(te_cm)), range(len(te_cm))].sum().round(2)

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm}%,Te:{te_cm}%')
        wandb.log(dict(class_l=loss.item(), class_tr=tr_cm, class_te=te_cm))
        loss.backward()
        optimizer.step()

    torch.save(classifier.cpu().state_dict(), Files.classifier_state)

    if os.path.exists(Files.classifier_state):
        classifier.load_state_dict(torch.load(Files.classifier_state))
    classifier.to(device)

    # le_classes = le.classes_.copy()
    # if len(le.classes_) not in np.unique(y_test):
    #     le_classes = le_classes[le_classes != "Sneeze"]  # prone to breaking, fix as this doesn't depend on where sneezes occur in the class vector

    # ConfusionMatrixDisplay(
    #     confusion_matrix(
    #         le.inverse_transform(y_test),
    #         le.inverse_transform(classifier(X_test).argmax(dim=1).detach().cpu()),
    #         normalize='true'
    #     ).round(2),
    #     display_labels=le.classes_.copy()
    # ).plot()
