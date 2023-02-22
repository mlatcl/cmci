import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

from utils import load_segments
from audio.audio_processing import get_spectrum_segment, preprocess_spectrum

from vae import VAE

plt.ion()

segment_list = load_segments(file_path='../../monkey_data/segments/June13th_segments.json')

segments = pd.DataFrame(segment_list)
segments.columns = ['start', 'end', 'file']
#"../../monkey_data/Banham/June13th/081014-013.wav"
# TODO
segments['id'] = segments.file.str.replace(r'../monkey_data/Banham/June..../', '').str.replace('.wav', '')


n_data, n_mels, new_segm_len = 10000, 64, 32 

Z = torch.load('Z.pth').numpy()
dim_reducer = VAE()
dim_reducer.load_state_dict(torch.load('sd.pth'))


# VAE projection

fig, (axa, axb, axc) = plt.subplots(1, 3)
axa.scatter(Z[:, 0], Z[:, 1], alpha=0.01)

record = []

while True:
    idx = np.random.choice(len(segment_list), 1)[0]

    start, end, f = segment_list[idx]
    print(f'-----------------\n')
    print(f'loading data: {segment_list[idx]}')
    S, freq, t = get_spectrum_segment(start, end, f, extension=0.35)
    
    mel_spec = preprocess_spectrum(S, n_mels, new_segm_len)

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
