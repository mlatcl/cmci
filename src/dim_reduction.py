
from tqdm import tqdm
import numpy as np
from verify_segments import get_segments, get_spectrum_segment
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram

import torch
from torchvision.transforms import Resize

from sklearn.decomposition import PCA

class DimReducer:
    def __init__(self):
        self.encoder = PCA(2, random_state=42)

    def fit(self, features):
        pass

    def latents(self, features):
        return self.encoder.fit_transform(features)

if __name__ == '__main__':
    plt.ion(); plt.style.use('seaborn-pastel')

    segment_list = get_segments()

    n_mels, new_segm_len = 16, 11
    specs_flattened = []
    for (start, end, f) in tqdm(segment_list):
        t, freq, S = get_spectrum_segment(start, end, f, extension=0.01)

        mel_spec = melspectrogram(S=S, n_mels=16)
        mel_spec = Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...]).numpy()[0, ...]
        specs_flattened.append(mel_spec.reshape(-1))

    specs_flattened = np.vstack(specs_flattened)

    reduced_dim = DimReducer().latents(specs_flattened)
    plt.scatter(reduced_dim[:, 0], reduced_dim[:, 1])