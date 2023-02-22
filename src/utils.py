import json
from scipy.io import wavfile as wav
from audio.audio_processing import get_spectrum
from functools import lru_cache


#### Segment Management

def load_segments(file_path='segments.json'):
    """
    Loads a segments.json file and returns an array with the information in tuples
    :returns: [(start, end, filename),]
    """
    with open(file_path, 'r') as fp:
        segments = json.load(fp)
    segment_list = []
    for f, segments in segments.items():
        sl = [(seg[0], seg[1], f) for seg in segments['segments']]
        segment_list.extend(sl)
    return segment_list 