
import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from callfinder import CallFinder
from audio.audio_processing import get_spectrum
import argparse
import json
import os
import time
import pandas as pd
from utils import preprocess_call_labels

if __name__ == '__main__':
    start = time.time()
    callFinder = CallFinder()

    parser = argparse.ArgumentParser(
                    prog = 'CallSegmenter',
                    description = 'CLI tool for taking in data folders of wav files and segmenting /saving them!')
    parser.add_argument('segments', help='Path (local) to the callfinder segments', default='../data/calls_for_ml/segments_ML_Test.json')           # positional argument
    parser.add_argument('labels', help='Path (local) to the training labels file', default='../data/calls_for_ml/Calls_ML.xlsx')           # positional argument
    args = parser.parse_args()
    segments_file = args.segments
    labels_file = args.labels

    ## Import CallFinder segments
    with open(segments_file, 'r') as fp:
        json_segments = json.load(fp)

    cols = ['start', 'end']
    segments = pd.DataFrame(columns=cols)
    for fname in json_segments:
        tmp_segs = pd.DataFrame(json_segments[fname]['segments'], columns=cols)
        tmp_segs['file'] = fname.split('/')[-1].replace('.wav', '')
        segments = pd.concat([segments, tmp_segs], axis=0)
    print(segments)

    ## Import Labelled Data
    labels_raw = pd.read_excel(labels_file)
    labels = preprocess_call_labels(labels_raw)

    print(labels)

    ## Combine the two datasets / call matching

    
    def finder(target, subjects, tolerance=0.1):
        """
        We loop over the ith target call and find calls from subject that match it.
        We take the center of a start / end of the call and see if it is within a certain amount of time (tolerance).
        """
        t_file, t_start, t_end = target.file, target.start, target.end

        subset = subjects.loc[subjects.file == t_file].copy()
        if len(subset) == 0: return np.nan

        target_center = (t_end + t_start) / 2
        subset['center'] = (subset['end']  + subset['start']) / 2
        subset['dif'] = (subset.center - target_center).abs()
        subset = subset.loc[subset.dif.abs() < tolerance, :]
        if len(subset) == 0: return np.nan

        return subset.reset_index().set_index('dif').sort_index().iloc[0]['index']

    # Call Matching
    labels['proposed_idx'] = [finder(labels.iloc[i], segments) for i in range(len(labels))]

    segments['label_idx'] = [finder(segments.iloc[i], labels) for i in range(len(segments))]

    total_labels = len(labels[(labels.file=='ML_Test') & (~labels.interference.isna())])
    labels = labels.loc[~labels.proposed_idx.isna()]
    idxs_that_are_calls = labels.proposed_idx.unique()
    segments.loc[idxs_that_are_calls, 'is_call'] = True


    print(segments)
    print(labels)

    ## Generate confusion matrix

    # How many true calls are we finding 
    # (how many do we miss)
    # How many false positives
    # True negative

    # segments are correct / incorrect
    # labels are found / not found
    true_segments = len(segments[segments.is_call == True])
    found_labels = len(labels)
    print("Segments correct / incorrect (FP): {} {:.2f}% / {} {:.2f}%".format(true_segments, float(true_segments) / float(len(segments)),  len(segments) - true_segments,  float(len(segments)- true_segments) / float(len(segments))))
    print("Labels found (TP) / not found (FN): {} {:.2f}% / {} {:.2f}%".format(found_labels, float(found_labels) / float(total_labels), total_labels - found_labels, float(total_labels - found_labels) / float(total_labels)))


