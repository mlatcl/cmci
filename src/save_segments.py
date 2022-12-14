
import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from callfinder import CallFinder
from audio.audio_processing import get_spectrum
import argparse
import json

import os


def download_s3_folder(bucket, s3_folder, local_dir=None):
    """
    Taken from: https://stackoverflow.com/a/62945526
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    print("Downloading S3 folder... {} / {}".format(bucket, s3_folder))
    if os.path.exists(os.path.join(local_dir)):
        print("Skipping the download because it's already downloaded")
        return
    for obj in bucket.objects.filter(Prefix=s3_folder):
        print("Looping for downloading S3 object {}".format(obj))
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
    
def save_segments(files_and_segments, file_name='segments.json', is_DDB=False):
    # TODO add DDB saving
    if not is_DDB:
        with open(file_name, 'w') as fp:
            json.dump(files_and_segments, fp)
    else:
        print("Please implement DDB saving!")

if __name__ == '__main__':

    start_time = 0
    callFinder = CallFinder()

    parser = argparse.ArgumentParser(
                    prog = 'CallSegmenter',
                    description = 'CLI tool for taking in data folders of wav files and segmenting /saving them!')
    parser.add_argument('input', help='The root directory with wav files and folders of them beneath it.')           # positional argument
    parser.add_argument('--output', help='The output directory to write the json file with files and call segment locations.')           # positional argument
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    is_s3 = False
    if input_dir.startswith("s3://"):
        is_s3 = True
        split_dir = input_dir[5:].split('/')
        bucket = split_dir[0]
        root_dir = split_dir[1:]
        download_s3_folder(bucket, root_dir, local_dir=root_dir)
    else:
        root_dir = input_dir

    walked_files = os.walk(root_dir) # [(dir, [dirs], [files]), (dir2, [dirs2], [files2])]
    files_and_segments = {}

    for (dir_name, directories, files) in walked_files:
        print("Walking directory {}".format(dir_name))
        for file in files:
            if file.endswith(".wav"):
                # TODO add a loop to process over all the start times / windows / length of the file?
                print("\tProcessing file {}".format(file))
                sampling_rate, audio = wav.read(os.path.join(dir_name, file))
                length_secs_audio = (audio.shape[0] // sampling_rate)
                segment_length = 10 if length_secs_audio > 60 else length_secs_audio - 1 # seconds, width of the spectrum we find_calls over
                # print("\tAUDIO {} {}".format(audio.shape, length_secs_audio))
                for curr_time in range(0, length_secs_audio-segment_length, segment_length):
                    # print("\t\tCurrent time {}".format(curr_time))
                    t, f, S = get_spectrum(start_time=curr_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)
                    segments, thresholded_spectrum, features, final_feature = callFinder.find_calls(S, f, t)
                    full_name = "s3://" + bucket + os.path.join(dir_name, file) if is_s3 else os.path.join(dir_name, file)
                    if full_name in files_and_segments:
                        files_and_segments[full_name] = {'segments': files_and_segments[full_name]['segments'] + segments.tolist()}
                    else:
                        files_and_segments[full_name] = {'segments': segments.tolist()}
        print()

    save_segments(files_and_segments)
    
    pass