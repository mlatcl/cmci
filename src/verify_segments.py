"""
This file samples 30 calls and generates images with the calls highlighted, 
for use in manual verification of how well the call identifier is doing.

"""


import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from callfinder import CallFinder
from audio.audio_processing import get_spectrum
import argparse
import json
import matplotlib.pyplot as plt
from functools import lru_cache

import os

def get_segments():
    with open("segments.json", 'r') as fp:
        segments = json.load(fp)
    segment_list = []
    for f, segments in segments.items():
        sl = [(seg[0], seg[1], f) for seg in segments['segments']]
        segment_list.extend(sl)
    return segment_list

@lru_cache(maxsize=20)
def load_audio_file(filename):
    return wav.read(filename)

def get_spectrum_segment(start, end, filename, extension=1.5):
        sampling_rate, audio = load_audio_file(filename)
        start = float(start)
        end = float(end)

        if start >= extension:
            start_extend = start - extension
        else:
            start_extend = start
        if end <= (len(audio) // sampling_rate) - extension:
            end_extend = end + extension
        else:
            end_extend = end
        return get_spectrum(
            start_time=start_extend,
            sampling_rate=sampling_rate,
            audio=audio,
            segment_length=end_extend-start_extend
        )

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


