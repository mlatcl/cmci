
from tqdm import tqdm
import numpy as np
from verify_segments import get_segments, get_spectrum_segment
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram

import torch
from torchvision.transforms import Resize

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from umap import UMAP

class DimReducer:
    def __init__(self):
        self.encoder = TSNE(2, random_state=42)

    def fit(self, features):
        pass

    def latents(self, features):
        return self.encoder.fit_transform(features)

if __name__ == '__main__':
    plt.ion(); plt.style.use('seaborn-pastel')

    segment_list = get_segments()

    n_mels, new_segm_len = 16, 11
    specs_flattened = []
    labels = []
    for (start, end, f) in tqdm(segment_list):
        t, freq, S = get_spectrum_segment(start, end, f, extension=0.01)

        mel_spec = melspectrogram(S=S, n_mels=16)
        mel_spec = Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...]).numpy()[0, ...]

        if mel_spec.shape == (n_mels, new_segm_len):
            specs_flattened.append(mel_spec.reshape(-1))
            if '_samples' in f:
                labels.append(f.replace('../data/', '').replace('_samples.wav', ''))
            else:
                labels.append('')

    specs_flattened = np.vstack(specs_flattened)
    labels = np.hstack(labels)

    reduced_dim = DimReducer().latents(specs_flattened)
    unique_labels = np.unique(labels)
    cols = ['blue', 'red', 'green', 'cyan', 'pink', 'black']
    for lb, c in zip(unique_labels, cols):
        plt.scatter(reduced_dim[labels == lb, 0], reduced_dim[labels == lb, 1], c=c, label=lb)
    plt.legend()