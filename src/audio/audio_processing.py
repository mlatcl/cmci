from sys import call_tracing
import numpy as np
from scipy.io import wavfile as wav
from scipy.signal import stft
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
plt.style.use('seaborn-pastel'); plt.ion()

def select_audio_snippet(audio, sr, start_time_secs, end_time_secs):
    batched_audio = audio[int(start_time_secs*sr):int(end_time_secs*sr)] # trill
    return batched_audio

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

def find_calls(S, f, hmm):
    s_min, s_max = S.min(), S.max()
    S = (S - s_min) / (s_max - s_min) # normalize
    S = S > 0.8 # maybe change this to quantile
    S[(f < 4e3) | (f > 7000), :] = 0.0

    hmm.fit(S.sum(axis=0).reshape(-1, 1))
    labels = []
    try:
        labels = hmm.predict(S.sum(axis=0).reshape(-1, 1))
        labels = np.diff(labels, prepend=0)

        starts = np.where(labels == 1)[0]
        ends = np.where(labels == -1)[0]

        if (len(starts) != len(ends)):
            if labels[np.where(labels != 0)[0][0]] == 1:
                # call identified at the end and doesn't finish
                ends = np.hstack([ends, len(labels) - 1])
            else:
                starts = np.hstack([0, starts])

        if (len(starts) != len(ends)) or (ends < starts).any():
            from IPython.core.debugger import set_trace; set_trace()
            print('find_calls has failed.')
        labels = np.vstack((starts, ends)).T
    except:
        from IPython.core.debugger import set_trace; set_trace()
    return labels

def save_spec(spectrum, file='spec.png'):
    plt.clf()
    plt.imshow(spectrum)
    plt.savefig(file)

if __name__ == '__main__':
    DATA_PATH = '../data/10MinSample.wav'

    # sampling rate, audio array
    sr, audio = wav.read(DATA_PATH)

    audio = audio[:, 0].astype('f') # mean across L, R channels
    start_time_secs = 33.5
    end_time_secs = 34.23
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

