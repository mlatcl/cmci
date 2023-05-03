
from sys import call_tracing
import numpy as np
from scipy.io import wavfile as wav
from scipy.signal import stft
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from functools import lru_cache
from callfinder import CallFinder
import torch
from torchvision.transforms import Resize

def split_audio(audio, sr, num_windows=None, window_length_secs=None):
    """
    Takes in a big numpy array and splits it!
    Specify either num_windows OR window_length
    :param num_windows:
    :param window_length: in seconds

    :returns: A list of audio snippets
    """
    audio_snips = []
    total_time = len(audio)
    if num_windows is not None:
        window_length = total_time // num_windows
    else:
        window_length = window_length_secs * sr

    i = 0
    for j in range(0, total_time, window_length):
        audio_snips.append(audio[i:j])
        i = j

    return audio_snips

def audio_to_spectrum(audio, sr, n_fft, hop_length, n_mels):
    #TODO? Subtract overall background noise for each section. Or how to detect background for preprocesssing????
    spectrum = np.log1p(melspectrogram(y=audio,
                            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
    return spectrum

def save_spec(spectrum, file='spec.png'):
    plt.clf()
    plt.imshow(spectrum)
    plt.savefig(file)

if __name__ == '__main__':
    DATA_PATH = '../data/10MinSample.wav'

    # sampling rate, audio array
    sr, audio = wav.read(DATA_PATH)

    audio = audio[:, 0].astype('f') # mean across L, R channels
    # batched_audio = audio[int(3.89*sr):int(4.46*sr)] # phi

    # nfft is window length. / 10 => 10 windows will be made
    n_fft = sr//10

    # hoplength is how far forward you go between windows (overlapping windows if hop < nfft)
    hop_length = sr//20

    #n_mels is number of freq channels
    n_mels = 64

    audio_snips = split_audio(audio, sr, window_length_secs=30)
    audio_spectrums = [audio_to_spectrum(snip, sr, n_fft, hop_length, n_mels) for snip in audio_snips]

    for i, spectrum in enumerate(audio_spectrums):
        call_locations = find_calls(spectrum, sr)
        save_spec(spectrum, file='../data/output_spectrograms/test_spec_{}.png'.format(i))

def get_spectrum(start_time, sampling_rate, audio, segment_length=10):
    """
    Start time is in seconds.
    Segement length is in seconds.
    """
    max_time = (len(audio)/sampling_rate)
    start_idx = max(0, min(int(sampling_rate * start_time), (max_time - segment_length - 1)*sampling_rate))
    end_idx = int(sampling_rate * (start_time + segment_length))
    f, t, spectrum = stft(audio[start_idx:end_idx], nperseg=sampling_rate//10, fs=sampling_rate)
    S = np.log(np.abs(spectrum) + 1e-10)
    return S, f, start_time + t

@lru_cache(maxsize=100)
def load_audio_file(filename):
    sr, audio = wav.read(filename)
    audio = audio[:, 0].astype('f')/1000 # take first channel, scale values down by 1000.
    return sr, audio

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
        S, f, t = get_spectrum(
            start_time=start_extend,
            sampling_rate=sampling_rate,
            audio=audio,
            segment_length=end_extend-start_extend
        )
        return S, f, t

def resize_func(mel_spec, n_mels, new_segm_len):
    a = torch.tensor(mel_spec)
    b = a[None, ...]
    c = Resize((n_mels, new_segm_len))(b)[None, ...]
    d = c.float()
    return d

def binarize(x):
    """
    Thresholding the spectrum here is peak finding.
    Threshold is 
    """
    x = CallFinder.threshold_spectrum(CallFinder.normalize_spectrum(x), 0.85, freq_to_ignore=5)
    return x

def preprocess_spectrum(S, n_mels, new_segm_len):
    mel_spec = melspectrogram(S=S, n_mels=n_mels)
    mel_spec = binarize(mel_spec)
    

    # Force the call to be of size (n_channels, n_mels, new_segm_len)\
    mel_spec = resize_func(mel_spec, n_mels, new_segm_len)
    return mel_spec