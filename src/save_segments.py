
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
    
def save_segments(files_and_segments, output_filepath='segments.json'):
    print("Saving segments to {}".format(output_filepath))
    with open(output_filepath, 'w') as fp:
        json.dump(files_and_segments, fp)

if __name__ == '__main__':

    start = time.time()
    callFinder = CallFinder()


    parser = argparse.ArgumentParser(
                    prog = 'CallSegmenter',
                    description = 'CLI tool for taking in data folders of wav files and segmenting /saving them!')
    parser.add_argument('input', help='The root directory with wav files and folders of them beneath it.')           # positional argument
    parser.add_argument('--output', help='The output directory and filename to write the json file with files and call segment locations.')           # positional argument
    args = parser.parse_args()
    input_dir = args.input
    output_filepath = args.output
    print("Time elapsed at {}: {}s".format("Beginning", time.time() - start))


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
    print("Time elapsed at {}: {:.2f}s".format("Files walked", time.time() - start))

    for (dir_name, directories, files) in walked_files:
        print("Walking directory {}".format(dir_name))
        for i, file in enumerate(files):
            if file.endswith(".wav"):
                sampling_rate, audio = wav.read(os.path.join(dir_name, file))
                print("Time elapsed at start of file {}: {:.2f}s".format(i, time.time() - start))
                audio = audio[:, 0].astype('f')/1000 # take first channel, scale values down by 1000.
                length_secs_audio = (audio.shape[0] // sampling_rate)
                segment_length = 10 if length_secs_audio > 60 else length_secs_audio - 1 # seconds, width of the spectrum we find_calls over
                # print("\tAUDIO {} {}".format(audio.shape, length_secs_audio))

                segment_times = range(0, length_secs_audio-segment_length, segment_length)
                print("\tProcessing file {} with {} {}s long segments".format(file, len(segment_times), segment_length))

                for curr_time in segment_times:
                    # print("Time elapsed start of time in file {}: {:.2f}s".format("curr_time", time.time() - start))
                    S, f, t = get_spectrum(start_time=curr_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)
                    # print("Time elapsed after {}: {:.2f}s".format("get_spectrum", time.time() - start))
                    segments, thresholded_spectrum, features, final_feature = callFinder.find_calls(S, f, t)
                    # print("Time elapsed after {}: {:.2f}s".format("find_calls", time.time() - start))
                    full_name = "s3://" + bucket + os.path.join(dir_name, file) if is_s3 else os.path.join(dir_name, file)
                    if full_name in files_and_segments:
                        files_and_segments[full_name] = {'segments': files_and_segments[full_name]['segments'] + segments.tolist()}
                    else:
                        files_and_segments[full_name] = {'segments': segments.tolist()}
                    # print()

    save_segments(files_and_segments, output_filepath=output_filepath)
    print("Time elapsed at {}: {:.2f}s".format("after saving final file", time.time() - start))
    
    pass