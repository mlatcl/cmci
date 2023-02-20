import json
from functools import lru_cache
from scipy.io import wavfile as wav
from audio.audio_processing import get_spectrum


#### Segment Management

def get_segments(file='segments.json'):
    with open(file, 'r') as fp:
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
