
import os, torch, wandb
import numpy as np
from tqdm import trange
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from librosa.feature import mfcc
from librosa.core.spectrum import stft
from scipy.io import wavfile as wav

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

softmax = torch.nn.Softmax(dim=1)

def process_file(f, start, end, n_fft_prop=1/3):
    sr, a = read_audio(f + '.wav')
    a = a[int(start * sr):int(end * sr)]

    S = np.abs(stft(a,
        n_fft=int(len(a) * n_fft_prop),
        hop_length=int(len(a) * n_fft_prop/2
    )))

    mel_features = mfcc(S=S, n_mfcc=20)
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)
    return mel_features.reshape(-1)

@lru_cache(maxsize=50)
def read_audio(f):
    sr, audio = wav.read(os.path.join(DATA_LOC, f))
    if len(audio.shape) == 2:
        return sr, audio.mean(axis=1)
    else:
        return sr, audio.astype(float)

class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.nnet = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.nnet(x)
        return softmax(x)

if __name__ == '__main__':

    from call_finder_rnn_simple import AudioDataset, Files, device
    DATA_LOC = Files.lb_data_loc

    data_loader = AudioDataset(device='cpu')
    calls = data_loader.labels.copy()

    calls = calls.loc[calls.end > calls.start].reset_index(drop=True)
    calls.loc[calls.call_type == 'Resonating Note', 'call_type'] = 'Resonate'

    # Reclassify call clusters
    # calls.loc[calls.call_type.isin(['Phee', 'Trill', 'Whistle']), 'call_type'] = 'LongCalls'
    # calls.loc[calls.call_type.isin(['Cheep', 'Chuck', 'Tsit']), 'call_type'] = 'ShortCalls'

    X = np.vstack([
        process_file(*calls.loc[i, ['file', 'start', 'end']])
        for i in calls.index
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    y = np.array(calls.call_type, dtype=str)
    le = LabelEncoder()
    le.fit(y)
    y_transformed = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)

    classifier = Classifier(X.shape[1], len(le.classes_))

    optimizer = torch.optim.Adam([
        dict(params=classifier.parameters(), lr=0.01),
    ])

    wandb.init(project="monke")

    losses = []; iterator = trange(10000, leave=False)
    for i in iterator:
        optimizer.zero_grad()

        y_prob = classifier(X_train)
        loss = -torch.distributions.Categorical(y_prob).log_prob(y_train).sum()

        tr_cm = confusion_matrix(y_train.cpu(), y_prob.argmax(dim=1).detach().cpu(), normalize='all')*100
        tr_cm = tr_cm[range(len(tr_cm)), range(len(tr_cm))].sum().round(2)

        te_cm = confusion_matrix(y_test, classifier(X_test).argmax(dim=1).detach().cpu(), normalize='all')*100
        te_cm = te_cm[range(len(te_cm)), range(len(te_cm))].sum().round(2)

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm}%,Te:{tr_cm}%')
        wandb.log(dict(class_l=loss.item(), class_tr=tr_cm, class_te=te_cm))
        loss.backward()
        optimizer.step()

    ConfusionMatrixDisplay(
        confusion_matrix(
            le.inverse_transform(y_test),
            le.inverse_transform(classifier(X_test).argmax(dim=1).detach().cpu()),
            normalize='true'
        ).round(2),
        display_labels=le.classes_.copy()
    ).plot()
