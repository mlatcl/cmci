
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torchvision.transforms import Resize

from utils import get_segments, get_spectrum_segment
from librosa.feature import melspectrogram

from vae import VAE

###############################################################
# VAE definition

if __name__ == '__main__':

    ###########################################################
    # Data prep

    # torch.manual_seed(42); np.random.seed(42)

    segment_list = get_segments(file='segments.json')

    # segments = pd.DataFrame(segment_list)
    # segments.columns = ['start', 'end', 'file']
    # segments['file_full'] = segments['file'].copy()
    # segments['file'] = segments.file.str.replace(r'../data/banham/June ..../', '').str.replace('.wav', '')
    # calls = pd.read_csv('../data/calls.csv')

    # def finder(i):
    #     file, time = calls.loc[i, 'File'], calls.loc[i, 'Time ']
    #     subset = segments.loc[segments.file == file].copy()
    #     subset['dif'] = (subset.start - time).abs()
    #     if len(subset) == 0: return np.nan
    #     subset = subset.loc[subset.dif.abs() < 0.1, :]
    #     if len(subset) == 0: return np.nan
    #     return subset.reset_index().set_index('dif').sort_index().iloc[0]['index']

    # calls['idx'] = [finder(i) for i in range(len(calls))]
    # calls = calls.loc[~calls.idx.isna()]

    # idxs_that_are_calls = calls.idx.unique()

    # segments.loc[idxs_that_are_calls, 'is_call'] = True
    # segments = segments.loc[np.random.choice(len(segments), len(segments), replace=False)].reset_index(drop=True)
    # segments = pd.concat([segments.loc[~segments.is_call.isna()], segments.loc[segments.is_call.isna()]]).reset_index(drop=True)

    n_data, n_mels, new_segm_len = 10000, 64, 32
    # specs = []
    # for i in trange(n_data):
    #     start, end, f = segments.loc[i, ['start', 'end', 'file_full']]
    #     t, freq, S = get_spectrum_segment(start, end, f, extension=0.1)

    #     mel_spec = melspectrogram(S=S, n_mels=n_mels)
    #     mel_spec = Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...])

    #     if mel_spec.shape == (1, n_mels, new_segm_len):
    #         specs.append(mel_spec)
    #     else:
    #         specs.append(torch.ones(1, n_mels, new_segm_len) * np.nan)

    def standardize(x): return x/5 + 1

    # X = standardize(torch.cat(specs, axis=0)[:, None, :, :]).cuda()
    # y = torch.tensor(segments.loc[:(n_data - 1), 'is_call'])

    # torch.save(X.cpu(), 'X.pth')
    # torch.save(y, 'y.pth')

    X = torch.load('X.pth').cuda()
    y = torch.load('y.pth').cuda()
    n_data = len(y)

    from callfinder import CallFinder
    def binarize(x):
        x = x[0].cpu().numpy()
        x = CallFinder.threshold_spectrum(CallFinder.normalize_spectrum(x), 0.85, freq_to_ignore=5)
        return torch.tensor(x)[None, None, ...].cuda().float()

    X = torch.cat([binarize(x) for x in tqdm(X)], axis=0)

    ###########################################################
    # Optim

    dim_reducer = VAE()
    # dim_reducer.load_state_dict(torch.load('sd.pth'))
    dim_reducer = dim_reducer.cuda()
    parameter_prior = torch.distributions.Normal(0, 0.1)
    latent_prior = torch.distributions.Normal(0, 1)

    optimizer = torch.optim.Adam([
        {'lr': 0.01, 'params': dim_reducer.parameters()},
    ])

    batch_size = 1000
    losses = []; bar = trange(10000, leave=False)
    for i in bar:
        optimizer.zero_grad()

        idx = np.random.choice(np.arange(n_data), batch_size)
        Zd, X_pred = dim_reducer(X[idx])
        obs_sd = torch.nn.Softplus(dim_reducer.log_obs_sd)

        elbo = torch.distributions.Normal(X_pred, obs_sd).log_prob(X[idx]).sum() - \
               torch.distributions.kl_divergence(Zd, latent_prior).sum()

        # for p in dim_reducer.parameters():
        #     elbo += parameter_prior.log_prob(p).sum()

        loss = -elbo/n_data

        loss.backward()
        optimizer.step()

        bar.set_description(f'n_elbo:{np.round(loss.item(), 2)}')
        losses.append(loss.item())

    Zd, _ = dim_reducer(X)
    Z = Zd.loc.detach().cpu()

    # torch.save(Z, 'Z.pth')
    # torch.save(dim_reducer.cpu().state_dict(), 'sd.pth')



    ###########################################################
    # Plotting

    Z = torch.load('Z.pth').numpy()

    plt.ion(); plt.style.use('seaborn-pastel')

    plt.scatter(Z[np.isnan(y.cpu()) == 1, 0], Z[np.isnan(y.cpu()) == 1, 1], alpha=0.01)
    plt.scatter(Z[np.isnan(y.cpu()) == 0, 0], Z[np.isnan(y.cpu()) == 0, 1], alpha=0.25)

    fig, (axa, axb, axc) = plt.subplots(1, 3)
    axa.scatter(Z[:, 0], Z[:, 1], alpha=0.01)

    record = []
    while True:
        idx = np.random.choice(len(segment_list), 1)[0]

        start, end, f = segment_list[idx]
        print(f'-----------------\n')
        print(f'loading data: {segment_list[idx]}')
        t, freq, S = get_spectrum_segment(start, end, f, extension=0.35)

        mel_spec = melspectrogram(S=S, n_mels=n_mels)
        mel_spec = standardize(Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...]))
        mel_spec = binarize(mel_spec).cuda()

        latent, _ = dim_reducer(mel_spec)
        latent = latent.loc.cpu().detach()

        axb.clear()
        axb.imshow(S, aspect='auto', origin='lower', cmap=plt.cm.binary)

        axc.imshow(mel_spec[0, 0].cpu(), aspect='auto', origin='lower')
        plt.draw()

        rect_start = int(0.35*S.shape[1] / ((end - start) + 2*0.35))
        rect_end = S.shape[1] - rect_start
        axb.add_patch(patches.Rectangle((rect_start, 0), rect_end - rect_start, len(freq), alpha=0.25))

        label = input("Is this a call? y/n/m/.:\n")
        col = dict(y='green', n='red', m='pink')[label]
        axa.scatter(latent[0, 0], latent[0, 1], c=col)

        record.append((
            segment_list[idx],
            label,
        ))

    from matplotlib import offsetbox

    plt.figure()
    ax = plt.subplot(aspect='equal')

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.tight_layout()

    ax.scatter(Z[:, 0], Z[:, 1], lw=0, s=40, alpha=0.01)

    # idx_to_plot = np.random.choice(np.arange(n_data), 20, replace=False)
    shown_images = Z[[0], :]
    for i in range(len(Z)):
        if np.square(Z[i] - shown_images).sum(axis=1).min() < 2:
        # if i not in idx_to_plot:
            continue
        plt.scatter(Z[i, 0], Z[i, 1], c='black', alpha=0.7)
        ax.add_artist(offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(X.cpu().numpy()[i, 0], cmap=plt.cm.autumn), Z[i, :]))
        shown_images = np.r_[shown_images, Z[[i], :]]
    plt.xticks([]), plt.yticks([])
