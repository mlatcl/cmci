
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torchvision.transforms import Resize

from utils import load_segments
from audio.audio_processing import get_spectrum_segment
from librosa.feature import melspectrogram
import time
import os

from audio.call_features import CallFeatures
from audio.audio_processing import preprocess_spectrum

from vae import VAE

###############################################################
# VAE definition

Softplus = torch.nn.Softplus()

if __name__ == '__main__':

    ###########################################################
    # Data prep

    torch.manual_seed(42); np.random.seed(42)
    # device="cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    MAX_CALL_DURATION = 0.5

    segment_list = load_segments(file_path='../../monkey_data/segments/June13th_segments.json')
    segment_list = [x for x in segment_list if x[1] - x[0] < MAX_CALL_DURATION]

    segments = pd.DataFrame(segment_list)
    segments.columns = ['start', 'end', 'file']
    #"../../monkey_data/Banham/June13th/081014-013.wav"
    # TODO
    segments['id'] = segments.file.str.replace(r'../monkey_data/Banham/June..../', '').str.replace('.wav', '')

    segments

    def finder(i, calls, segments):
        file, time = calls.loc[i, 'File'], calls.loc[i, 'Time ']
        subset = segments.loc[segments.file == file].copy()
        subset['dif'] = (subset.start - time).abs()
        if len(subset) == 0: return np.nan
        subset = subset.loc[subset.dif.abs() < 0.1, :]
        if len(subset) == 0: return np.nan
        return subset.reset_index().set_index('dif').sort_index().iloc[0]['index']

    calls = pd.read_csv('../../monkey_data/labelled_calls/calls.csv')

    # Call Matching
    calls['idx'] = [finder(i, calls, segments) for i in range(len(calls))]
    calls = calls.loc[~calls.idx.isna()]
    idxs_that_are_calls = calls.idx.unique()
    segments.loc[idxs_that_are_calls, 'is_call'] = True

    # Data shuffling and sampling
    segments = segments.loc[np.random.choice(len(segments), len(segments), replace=False)].reset_index(drop=True) # shuffle the data
    segments = pd.concat([segments.loc[~segments.is_call.isna()], segments.loc[segments.is_call.isna()]]).reset_index(drop=True) # bring the labelled data to the front so that it is always selected

    # n_data number of sampled calls
    # n_mels mel frequency buckets
    # new_segm_len is the forced size of input to VAE.  
    n_data, n_mels, new_segm_len = 10000, 64, 32 

    # only 1 right now
    n_channels = 1

    def preprocess_segments():
        # Make inputs that go into the VAE, (X, y)
        specs = [] # list of mel spectrograms, shape = (n_data, n_mels, new_segm_len)
        features = []



        # TODO this is slow.
        thebest_time = time.time()
        # print("Time at {} is {:.2f}".format("start", time.time() - thebest_time))

        for i in trange(n_data):
            start, end, f = segments.loc[i, ['start', 'end', 'file']]
            # from IPython.core.debugger import set_trace; set_trace()
            S, freq, t = get_spectrum_segment(start, end, f, extension=0.1)
            # print("Computing length of t indexes", t, t.shape, np.diff(t))

            mel_spec = preprocess_spectrum(S, n_mels, new_segm_len)


            if mel_spec.shape == (n_channels, n_mels, new_segm_len):
                specs.append(mel_spec)
            else:
                specs.append(torch.ones(1, n_channels, n_mels, new_segm_len) * np.nan)

            S, freq, t = get_spectrum_segment(start, end, f, extension=0.001)
            
            cf = CallFeatures(S, freq, t)
            feature_tuple = np.hstack((cf.maximal_power_frequencies(), cf.max_freq(), cf.duration(), cf.maximal_power(),))
            features.append(torch.tensor(feature_tuple)[None, ...])

        # def standardize(x): return x/5 + 1 # this roughly works for mel-spectrograms? 

        X = torch.cat(specs, axis=0).cpu().float()
        X_features = torch.cat(features, axis=0).cpu().float()
        # X = standardize((X)) # Uncomment this if we're using the raw or mel spectrograms
        y = torch.tensor(segments.loc[:(n_data - 1), 'is_call'])

        return X, X_features, y




    # Load data
    if os.path.exists('X.pth') and os.path.exists('y.pth') and os.path.exists('X_features.pth'):
        X = torch.load('X.pth').to(device)
        X_features = torch.load('X_features.pth').to(device)
        y = torch.load('y.pth').to(device)
    else:
        X, X_features, y = preprocess_segments()
        torch.save(X, 'X.pth')
        torch.save(X_features, 'X_features.pth')
        torch.save(y, 'y.pth')
    n_data = len(y)

    print(X_features)

    ###########################################################
    # Optim

    dim_reducer = VAE()
    # dim_reducer.load_state_dict(torch.load('sd.pth'))
    dim_reducer = dim_reducer.to(device)
    parameter_prior = torch.distributions.Normal(0, 0.1)
    latent_prior = torch.distributions.Normal(0, 1)

    optimizer = torch.optim.Adam([
        {'lr': 0.01, 'params': dim_reducer.parameters()},
    ])

    batch_size = 1000
    n_iters = 1000
    losses = []; bar = trange(n_iters, leave=False)
    for i in bar:
        optimizer.zero_grad()

        idx = np.random.choice(np.arange(n_data), batch_size)

        Zd, X_pred = dim_reducer(X[idx])
        obs_sd = Softplus(dim_reducer.log_obs_sd)

        elbo = torch.distributions.Normal(X_pred, obs_sd).log_prob(X[idx]).sum() - \
               torch.distributions.kl_divergence(Zd, latent_prior).sum()

        loss = -elbo/n_data

        loss.backward()
        optimizer.step()

        bar.set_description(f'n_elbo:{np.round(loss.item(), 2)}')
        losses.append(loss.item())

    Zd, _ = dim_reducer(X)
    Z = Zd.loc.detach().cpu()

    torch.save(Z, 'Z.pth')
    torch.save(dim_reducer.cpu().state_dict(), 'sd.pth')

    dim_reducer.to(device)
