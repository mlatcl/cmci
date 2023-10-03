
import os, torch
import numpy as np
from tqdm import trange
from call_finder_rnn_simple import \
    load_audio, FEATURIZER, Files, N_MELS, Classifier, device, \
    CallFinder as CallFinderRNN

class Files_SL(Files):
    unlb_data_loc = Files.lb_data_loc.replace('labelled', 'unlabelled')
    state_dict = '../data/Calls for ML/smol_sl_sd.pth'

class SSLData(torch.utils.data.Dataset):
    def __init__(self):
        self.audio_files = os.listdir(Files_SL.unlb_data_loc)
        self.features = []
        for file in self.audio_files:
            audio = load_audio(os.path.join(Files_SL.unlb_data_loc, file)).to(device)
            self.features.append(FEATURIZER(audio).T)

    def __len__(self):
        return np.inf

    def __getitem__(self, index, segm_len=200, num_sample_per_file=100):
        sampled_features = []
        for feature in self.features:
            idx_max = feature.shape[0] - segm_len
            for _ in range(num_sample_per_file * int(np.ceil(feature.shape[0]/1e5))):
                if idx_max > 0:
                    sample_idx = np.random.choice(idx_max)
                    sampled_features.append(
                        feature[sample_idx:(sample_idx + segm_len), :][None, ...]
                    )
        return torch.cat(sampled_features, axis=0)

class CallFinder(CallFinderRNN):
    def __init__(self):
        super().__init__()
        self.classifier = Classifier(N_MELS, num_lstm=6)
        self.classifier.load_state_dict(torch.load(Files_SL.state_dict))
        self.classifier.to(device)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion(); plt.style.use('seaborn-v0_8-pastel')

    np.random.seed(42); torch.manual_seed(42)

    model_v1 = Classifier(N_MELS)
    model_v1.load_state_dict(torch.load(Files.state_dict))
    model_v1.to(device)

    dataloader = SSLData()

    model_v2 = Classifier(N_MELS, num_lstm=6).to(device)
    optimizer = torch.optim.Adam(model_v2.parameters(), lr=0.005)

    data_iterator = iter(dataloader); losses = []

    for i in (iterator := trange(1800, leave=False)):
        optimizer.zero_grad()

        X = next(data_iterator)

        y_true = model_v1(X)
        y_pred = model_v2(X)

        loss = torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(y_pred),
            torch.distributions.Bernoulli(y_true)
        ).sum()

        losses.append(loss.item())
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    torch.save(model_v2.cpu().state_dict(), Files_SL.state_dict)
    model_v2.to(device)

    X_orig = FEATURIZER(
        load_audio(Files.lb_data_loc + 'Blackpool_Combined_FINAL.wav').to(device)
    ).T[None, :1000, :]

    plt.plot(model_v1(X_orig)[0].detach().cpu(), label='v1')
    plt.plot(model_v2(X_orig)[0].detach().cpu(), label='v2')
    plt.legend()
    plt.tight_layout()
