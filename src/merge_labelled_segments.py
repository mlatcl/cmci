
import os
import numpy as np
from scipy.io import wavfile as wav

CLASSES = ['Chirp', 'JaggedTrill', 'Moan', 'Phee', 'Trill', 'Tsit']

if __name__ == '__main__':
    sr = 44100
    silence = np.zeros((int(sr * 0.2), 2), dtype=np.int16)

    data_path = 'labelled_data'
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f != 'merged']

    merged_data_path = os.path.join(data_path, 'merged')
    if not os.path.exists(merged_data_path):
        os.mkdir(merged_data_path)

    for class_name in CLASSES:
        wav_concat = np.vstack([ \
            np.vstack([wav.read(f)[1], silence]) \
            for f in files if class_name in f \
        ])
        wav.write(os.path.join(merged_data_path, f'{class_name}_samples.wav'), 44100, wav_concat)
