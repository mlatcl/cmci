"""
This file samples 30 calls and generates images with the calls highlighted, 
for use in manual verification of how well the call identifier is doing.

"""


import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wavfile

import matplotlib.pyplot as plt
from utils import get_segments, get_spectrum_segment

import os

if __name__ == '__main__':
    segment_list = get_segments()

    print(len(segment_list))
    idx_samples = np.random.choice(range(len(segment_list)), 30)
    sampled_segments = np.array(segment_list)[idx_samples, :]
    print(sampled_segments)

    for (start, end, f) in sampled_segments:
        t, freq, S = get_spectrum_segment(start, end, f)

        spectrum_fig = px.imshow(S, aspect='auto', x=t, y=freq, origin='lower',
            labels=dict(x='Time (sec)', y='Freq (Hz)'))
        spectrum_fig.add_shape(x0=start, x1=end, y0=freq[0], y1=freq[-1], opacity=0.25, fillcolor="Green")

        
        spectrum_fig.write_image("../tmp/v2/{}-{}-{}.png".format(int(start*1000), int(end*1000), f.split('/')[-1].split('.wav')[0]))


