
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.transforms import Resize

from verify_segments import get_segments, get_spectrum_segment
from librosa.feature import melspectrogram

###############################################################
# VAE definition

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 2, 1)

class VAE(nn.Module):
    """ Based on sksq96/pytorch-vae """
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.mu = nn.Linear(512, latent_dim)
        self.log_sd = nn.Linear(512, latent_dim)
        self.log_obs_sd = torch.nn.Parameter(torch.ones(1))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_log_sd = self.mu(h), self.log_sd(h)
        z_dist = torch.distributions.Normal(z_mu, z_log_sd.exp())
        z = z_dist.rsample()
        return z_mu, self.decoder(z)

if __name__ == '__main__':

    ###########################################################
    # Data prep

    # segment_list = get_segments()

    # n_mels, new_segm_len = 32, 16
    # specs = []
    # labels = []
    # for (start, end, f) in tqdm(segment_list):
    #     t, freq, S = get_spectrum_segment(start, end, f, extension=0.01)

    #     mel_spec = melspectrogram(S=S, n_mels=n_mels)
    #     mel_spec = Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...])

    #     if mel_spec.shape == (1, n_mels, new_segm_len):
    #         specs.append(mel_spec)
    #         if '_samples' in f:
    #             labels.append(f.replace('../data/', '').replace('_samples.wav', ''))
    #         else:
    #             labels.append('')

    # def standardize(x): return x/5 + 1

    # X = standardize(torch.cat(specs, axis=0)[:, None, :, :]).cuda()
    # labels = np.hstack(labels)

    # torch.save(X.cpu(), 'X.pth')
    # torch.save(labels, 'labels.pth')

    X = torch.load('X.pth')
    labels = torch.load('labels.pth')

    ###########################################################
    # Optim

    dim_reducer = VAE()
    dim_reducer.load_state_dict(torch.load('sd.pth'))
    dim_reducer = dim_reducer.cuda()
    parameter_prior = torch.distributions.Normal(0, 0.1)

    optimizer = torch.optim.Adam([
        {'lr': 0.01, 'params': dim_reducer.parameters()},
    ])

    losses = []; bar = trange(10000, leave=False)
    for i in bar:
        optimizer.zero_grad()

        Z, X_pred = dim_reducer(X)
        obs_sd = dim_reducer.log_obs_sd.exp()

        loss = -torch.distributions.Normal(X_pred, obs_sd).log_prob(X).sum()
        for p in dim_reducer.parameters():
            loss -= parameter_prior.log_prob(p).sum()

        loss.backward()
        optimizer.step()

        bar.set_description(f'n_elbo:{np.round(loss.item(), 2)}')

    # torch.save(Z.cpu().detach(), 'Z.pth')
    # torch.save(dim_reducer.cpu().state_dict(), 'sd.pth')

    ###########################################################
    # Plotting

    Z = torch.load('Z.pth')

    plt.ion(); plt.style.use('seaborn-pastel')

    reduced_dim = Z.detach().cpu().numpy()
    # unique_labels = np.unique(labels)
    # cols = ['blue', 'red', 'green', 'cyan', 'pink', 'black']
    # for lb, c in zip(unique_labels, cols):
    #     plt.scatter(reduced_dim[labels == lb, 0], reduced_dim[labels == lb, 1], c=c, label=lb, alpha=0.01)
    # plt.legend()

    from matplotlib import offsetbox

    plt.figure()
    ax = plt.subplot(aspect='equal')

    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.tight_layout()

    ax.scatter(reduced_dim[:, 0], reduced_dim[:, 1], lw=0, s=40, alpha=0.01)

    shown_images = reduced_dim[[0], :]
    for i in range(len(reduced_dim)):
        if np.square(reduced_dim[i] - shown_images).sum(axis=1).min() < 0.75:
            continue
        plt.scatter(reduced_dim[i, 0], reduced_dim[i, 1], c='black', alpha=0.7)
        ax.add_artist(offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(X[i, 0], cmap=plt.cm.autumn), reduced_dim[i, :]))
        shown_images = np.r_[shown_images, reduced_dim[[i], :]]
    plt.xticks([]), plt.yticks([])
