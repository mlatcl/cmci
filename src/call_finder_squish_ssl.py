
import os, wandb, torch, warnings
import numpy as np
from tqdm import trange
from call_finder_rnn_simple import \
    CallFinder as CallFinderRNN, Classifier as RNN, \
    Files as FilesRNN

from call_finder_squish import CallFinder as CallFinderV1, \
    Files, load_audio, FEATURIZER, N_MELS, device, AudioDataset

from sklearn.metrics import confusion_matrix
# import wandb

class Files_SL(Files):
    unlb_data_loc = Files.lb_data_loc.replace('labelled', 'unlabelled')
    # state_dict = '../data/Calls for ML/smol_sl_sd.pth'

class SSLData(torch.utils.data.Dataset):
    def __init__(self):
        self.audio_files = os.listdir(Files_SL.unlb_data_loc)
        self.features = []
        for file in self.audio_files:
            audio = load_audio(os.path.join(Files_SL.unlb_data_loc, file))
            self.features.append(FEATURIZER.cpu()(audio).T)
        FEATURIZER.to(device)

    def __getitem__(self, index, segm_len=200, num_sample_per_file=30, num_files=4):
        sampled_features = []
        files_to_sample = np.random.choice(len(self.features), num_files)
        files_to_sample = [self.features[i] for i in files_to_sample]
        for feature in files_to_sample:
            idx_max = feature.shape[0] - segm_len
            for _ in range(num_sample_per_file * int(np.ceil(feature.shape[0]/1e5))):
                if idx_max > 0:
                    sample_idx = np.random.choice(idx_max)
                    sampled_features.append(
                        feature[sample_idx:(sample_idx + segm_len), :][None, ...].to(device)
                    )
        return torch.cat(sampled_features, axis=0)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion(); plt.style.use('seaborn-v0_8-pastel')

    np.random.seed(42); torch.manual_seed(42)

    rnn_simple = RNN(N_MELS)
    rnn_simple.load_state_dict(torch.load(FilesRNN.state_dict))
    rnn_simple.to(device)
    rnn_simple.eval()

    dataloader = SSLData()

    FEATURIZER = FEATURIZER.cpu()
    dataloader_squish = AudioDataset(device='cpu')

    cf_v1 = CallFinderV1()
    cf_v1.classifier.eval()

    model_v2 = CallFinderV1().classifier

    X_train, y_train, X_test, y_test = torch.load(Files.train_data)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)

    X_test_2 = dataloader_squish.featurizer(dataloader_squish.audio['ML_Test_3']).T[None, ...].to(device)
    y_test_2 = dataloader_squish.label_ts['ML_Test_3'].cpu().numpy()
    FEATURIZER = FEATURIZER.to(device)

    idx_to = (dataloader_squish.ts['Blackpool_Combined_FINAL'].cpu() - 10).abs().argmin().item()
    idx_to = int(len(dataloader_squish.audio['Blackpool_Combined_FINAL']) * idx_to / len(dataloader_squish.ts['Blackpool_Combined_FINAL']))

    optimizer = torch.optim.Adam(model_v2.parameters(), lr=0.001)

    data_iterator = iter(dataloader); losses = []
    wandb.init(project="monke")
    for i in (iterator := trange(10000, leave=False)):
        optimizer.zero_grad()

        #### comp against RNN
        X = next(data_iterator)

        with torch.no_grad():
            y_hat_rnn = rnn_simple(X)
        y_pred = model_v2(X)

        loss = torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(1 - y_pred[:, :, 0]),
            torch.distributions.Bernoulli(y_hat_rnn)
        ).sum()

        #### comp against v1

        # with torch.no_grad():
        #     y_hat_v1 = cf_v1.classifier(X)

        # loss += torch.distributions.kl_divergence(
        #     torch.distributions.Categorical(y_pred),
        #     torch.distributions.Categorical(y_hat_v1)
        # ).sum()

        #### comp against train data

        idx = np.random.choice(len(y_train), len(X))
        y_pred_on_train = model_v2(X_train[idx])
        loss -= torch.distributions.Categorical(y_pred_on_train).log_prob(y_train[idx]).sum()

        if i % 100 == 0:
            tr_cm = confusion_matrix(y_train[idx].reshape(-1).cpu(), y_pred_on_train.argmax(dim=-1).reshape(-1).detach().cpu(), normalize='all').round(3)*100
            tr_cm = tr_cm[range(len(tr_cm)), range(len(tr_cm))].sum().round(2)

            pred = model_v2(X_test).argmax(dim=-1).detach().cpu().reshape(-1)
            cm = confusion_matrix(y_test, pred, normalize='all').round(3)*100
            cm = cm[range(len(cm)), range(len(cm))].sum().round(2)

            cm_rn = confusion_matrix(y_test, pred, normalize='true').round(3)*100
            cm_rn = cm_rn[range(len(cm_rn)), range(len(cm_rn))].sum().round(2)

            pred_2 = model_v2(X_test_2).argmax(dim=-1).detach().cpu().reshape(-1)
            cm_2 = confusion_matrix(y_test_2, pred_2, normalize='all').round(3)*100
            cm_2 = cm_2[range(len(cm_2)), range(len(cm_2))].sum().round(2)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                num_calls = len(CallFinderV1(model_v2, dataloader_squish.le).find_calls_rnn(dataloader_squish.audio['Blackpool_Combined_FINAL'][:idx_to])[0])

        losses.append(loss.item())

        iterator.set_description(f'L:{np.round(loss.item(), 2)},Tr:{tr_cm},Te:{cm},Te2:{cm_2}')
        wandb.log(dict(ssl_l=loss.item(), ssl_tr=tr_cm, ssl_te=cm, ssl_te_rn=cm_rn, ssl_num_bp=num_calls, ssl_te_mlt3=cm_2))
        loss.backward()
        optimizer.step()

    # torch.save(model_v2.cpu().state_dict(), 'xx.pth')
    # model_v2.to(device)

    # X_orig = FEATURIZER(
    #     load_audio(Files.lb_data_loc + 'Blackpool_Combined_FINAL.wav').to(device)
    # ).T[None, :1000, :]

    # plt.plot(rnn_simple(X_orig)[0].detach().cpu(), label='v1')
    # plt.plot(1 - model_v2(X_orig)[0, :, 0].detach().cpu(), label='v2')
    # plt.legend()
    # plt.tight_layout()
