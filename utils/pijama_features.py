import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random

import config
from utilities import create_folder, traverse_folder, float32_to_int16
from pathlib import Path


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def pack_audio_files_to_hdf5(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = 16000 # config.sample_rate
    sample_duration_sec = 15 # config.clip_samples
    clip_samples = sample_rate * sample_duration_sec
    classes_num = 120 #config.classes_num

    metadata_path = Path(dataset_dir, "pijama.csv")
    with open(metadata_path) as f:
        r = csv.DictReader(f)
        metadata = [row for row in r]

    artists_path = Path(workspace, "artists.txt")
    with open(artists_path) as f:
        artists = sorted(f.read().splitlines())

    classes_num = len(artists)
    lb_to_idx = {lb: idx for idx, lb in enumerate(artists)}

    # compute hdf5 parameters
    audios_num = 0
    for track in metadata:
        performance_start_sec = float(track['performance_start_sec'])
        performance_end_sec = float(track['performance_end_sec'])
        performance_duration = performance_end_sec - performance_start_sec
        num_samples_per_track = performance_duration // sample_duration_sec
        audios_num += num_samples_per_track

    # Paths
    packed_hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S80')

        hf.create_dataset(
            name='waveform',
            shape=(audios_num, clip_samples),
            dtype=np.int16)

        hf.create_dataset(
            name='target',
            shape=(audios_num, classes_num),
            dtype=np.float32)

        hf.create_dataset(
            name='split',
            shape=(audios_num,),
            dtype='S20')

        n = 0
        for track in metadata:
            audio_path = Path(dataset_dir, track['mp3_filepath'].replace('/audio/', '/resampled/'))
            split = track['split']
            artist = track['artist']
            target = lb_to_idx[artist]
            (audio, fs) = librosa.core.load(audio_path, sr=None, mono=True)

            performance_start_sec = float(track['performance_start_sec'])
            performance_end_sec = float(track['performance_end_sec'])
            performance_duration = performance_end_sec - performance_start_sec
            num_samples = int(performance_duration // sample_duration_sec)
            performance_offset = int(performance_start_sec * sample_rate)

            readable_name_base = track["title"] + "_" + track["artist"] + "_" + track["album"]
            readable_name_base = "".join(readable_name_base.lower().split())

            for i in range(num_samples):
                audio_name = f"{readable_name_base}_{i}"
                sample = audio[performance_offset + i*clip_samples:performance_offset + (i+1)*clip_samples]
                sample = pad_truncate_sequence(sample, clip_samples)

                hf['audio_name'][n] = audio_name.encode()
                hf['target'][n] = to_one_hot(target, classes_num)
                hf['waveform'][n] = float32_to_int16(sample)
                hf['split'][n] = split

                n += 1

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')
