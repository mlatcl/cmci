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


def preprocess_call_labels(calls_og, keep_only_conures=True):
    calls = calls_og.copy()
    calls.columns = [c.lower().replace(' ', '_') for c in calls.columns]
    if keep_only_conures:
        calls = calls.loc[~calls.call_type.isna() | (calls.interference == 'Conure')].reset_index(drop=True) #drop conure calls?
    calls.loc[calls.call_type.isna(), 'call_type'] = 'interference' # set all unknown call types to interference
    calls = calls.loc[calls.start < calls.end].reset_index(drop=True)
    calls['call_type'] = calls.call_type.apply(lambda r: r.split(' ')[0])
    return calls