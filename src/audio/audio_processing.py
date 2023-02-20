
from sys import call_tracing
import numpy as np
from scipy.io import wavfile as wav
from scipy.signal import stft
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
plt.style.use('seaborn-pastel'); plt.ion()

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

def find_calls_old(spectrum, sr):
    call_locations = [] # (start, stop)
    call_locations.append((int(3.89*sr),int(4.46*sr))) # phi
    # TODO be smart.

    return call_locations

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
    start_idx = min(int(sampling_rate * start_time), (max_time - segment_length - 1)*sampling_rate)
    end_idx = int(sampling_rate * (start_time + segment_length))
    f, t, spectrum = stft(audio[start_idx:end_idx], nperseg=sampling_rate//10, fs=sampling_rate)
    return np.log(np.abs(spectrum) + 1e-10), f, start_time + t