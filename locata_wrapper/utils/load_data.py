# -*- coding: utf-8 -*-
# Copyright 2019 Waseda University (Nelson Yalta)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import Namespace
import glob
import logging
import numpy as np
import os
import pandas as pd
import python_speech_features  # NOQA
import soundfile
import sys

from locata_wrapper.utils.shared import ElapsedTime
from locata_wrapper.utils.shared import wrapToPi

from scipy.signal import resample

from matplotlib import pyplot as plt


def cart2pol(cart):
    """cart2pol

    Cartesian to spherical transformation for LOCATA coordinate system
    Inputs:
        x:      Cartesian x-position [m]
        y:      Cartesian y-position [m]
        z:      Cartesian z-position [m]

    Outputs:
        az:     Azimuth [rad]
        el:     Elevation [rad]
        rad:      Radius [m]
    """

    pol = np.zeros(cart.shape)
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    # radius
    pol[:, 2] = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2)
    # elev
    pol[:, 1] = np.arccos(z / pol[:, 2])
    # azimuth
    pol[:, 0] = wrapToPi(np.arctan2(y, x) - (np.pi / 2))
    return pol


def load_wav(fnames, obj_type):
    obj = Namespace()
    obj.data = dict()
    for this_wav in fnames:
        # Load data:
        data, fs = soundfile.read(this_wav)

        # Array name:
        this_obj = os.path.basename(this_wav).replace('.wav', '')
        this_obj = this_obj.replace('{}_'.format(obj_type), '')

        # Load timestamps:
        _txt_table = this_wav.replace('{}.wav'.format(this_obj),
                                      'timestamps_{}.txt'.format(this_obj))
        txt_table = np.loadtxt(_txt_table, delimiter='\t', skiprows=1).T

        # Copy to namespace:
        obj.fs = fs
        obj.data[str(this_obj)] = data
        obj.time = txt_table
    return obj


def load_txt(fnames, obj_type):
    obj = Namespace()
    obj.data = dict()
    for this_txt in fnames:
        # Load data:
        txt_table = pd.read_csv(this_txt, sep='\t', header=0)
        _time = txt_table[['year', 'month', 'day',
                           'hour', 'minute', 'second']]
        _pos = txt_table[['x', 'y', 'z']].values.T
        _ref = txt_table[['ref_vec_x', 'ref_vec_y', 'ref_vec_z']].values.T
        _rot_1 = txt_table[['rotation_11', 'rotation_12', 'rotation_13']].values
        _rot_2 = txt_table[['rotation_21', 'rotation_22', 'rotation_23']].values
        _rot_3 = txt_table[['rotation_31', 'rotation_32', 'rotation_33']].values
        _rot = np.stack([_rot_1, _rot_2, _rot_3], axis=0)

        mics = list(set([x.split('_')[0] for x in txt_table if 'mic' in x]))
        if len(mics) > 0:
            for i in range(len(mics)):
                _lbl = ['mic{}_{}'.format(i + 1, x) for x in ['x', 'y', 'z']]
                _data = txt_table[_lbl].values.T
                if i == 0:
                    _mic = np.zeros((3, _data.shape[1], len(mics)))
                _mic[:, :, i] = _data
        else:
            _mic = None

        # Array name:
        this_obj = os.path.basename(this_txt).replace('.txt', '')
        this_obj = this_obj.replace('{}_'.format(obj_type), '')

        # Copy to namespace:
        obj.time = pd.to_datetime(_time)
        obj.data[str(this_obj)] = Namespace(
            position=_pos, ref_vec=_ref, rotation=_rot,
            mic=_mic)
    return obj


def LoadData(this_array, log=logging, is_dev=True):
    """loads LOCATA csv and wav data

    Inputs:
        this_array:     Directory name containing array data

    Outputs:
        audio_array:    Structure containing audio data recorded at each of the arrays
        audio_source:   Structure containing clean speech data
        position_array:   Structure containing positional information of each of the arrays
        position_source:  Structure containing positional information of each source
        required_time:    Structure containing the timestamps at which participants must provide estimates
    """

    # Time vector:
    txt_array = pd.read_csv(os.path.join(this_array, 'required_time.txt'),
                            sep='\t')
    _time = pd.to_datetime(txt_array[['year', 'month', 'day',
                                      'hour', 'minute', 'second']])
    _valid = np.array(txt_array['valid_flag'].values, dtype=np.bool)
    required_time = Namespace(time=_time, valid_flag=_valid)

    # Audio files:
    wav_fnames = glob.glob(os.path.join(this_array, '*.wav'))

    audio_array_idx = [x for x in wav_fnames if 'audio_array' in x]
    if is_dev:
        audio_source_idx = [x for x in wav_fnames if 'audio_source' in x]
        if len(audio_array_idx) + len(audio_source_idx) == 0:
            log.error(f'Unexpected audio file in folder {this_array}')
            sys.exit(1)
    else:
        if len(audio_array_idx) == 0:
            log.error(f'Unexpected audio file in folder {this_array}')
            sys.exit(1)

    # Audio array data
    audio_array = load_wav(audio_array_idx, 'audio_array')

    # Audio source data:
    if is_dev:
        audio_source = load_wav(audio_source_idx, 'audio_source')
        audio_source.NS = len(audio_source.data)
    else:
        audio_source = None

    # Position source data:
    txt_fnames = glob.glob(os.path.join(this_array, '*.txt'))
    if is_dev:
        position_source_idx = [x for x in txt_fnames if 'position_source' in x]
        position_source = load_txt(position_source_idx, 'position_source')
    else:
        position_source = None

    # Position array data:
    position_array_idx = [x for x in txt_fnames if 'position_array' in x]
    position_array = load_txt(position_array_idx, 'position_array')

    # Outputs:
    return audio_array, audio_source, position_array, position_source, required_time


def GetTruth(array_dir, position_array, position_source, required_time, is_dev=True, log=logging):
    """GetLocataTruth

    creates Namespace containing OptiTrac ground truth data for and relative to the specified array

    Inputs:
        array_name:       String containing array name: 'eigenmike', 'dicit', 'benchmark2', 'dummy'
        position_array:   Structure containing array position data
        position_source:  Structure containing source position data
        required_time:    Vector of timestamps at which ground truth is required
        is_dev:           If 0, the evaluation database is considered and the
                          development database otherwise.

    Outputs:
        truth:            Namespace containing ground truth data
                          Positional information about the sound sources are
                          only returned for the development datbase
                          (is_dev = 1).
    """
    this_array = os.path.basename(array_dir)
    truth = Namespace()
    fs = 48e3

    # Specified array
    truth.array = position_array.data[this_array]
    truth.timestamps = ElapsedTime(required_time.time)  # 120Hz  t_stamps_opti
    for field in truth.array.__dict__:
        _new_value = getattr(truth.array, field)[:, required_time.valid_flag]
        setattr(truth.array, field, _new_value)

    # Source
    if is_dev:
        frames = int(np.sum(required_time.valid_flag))
        truth.source = position_source.data
        # All sources for this recording

        for src_idx in truth.source:
            for field in truth.source[src_idx].__dict__:
                _new_value = getattr(truth.source[src_idx], field)
                if _new_value is not None:
                    setattr(truth.source[src_idx], field, _new_value[:, required_time.valid_flag])

            # Azimuth and elevation relative to microphone array
            h_p = truth.source[src_idx].position - truth.array.position

            # Apply rotation / translation of array to source
            pol_pos = np.squeeze(np.matmul(truth.array.rotation.transpose(1, 2, 0), h_p.T[:, :, None]))

            # Returned in azimuth, elevation & radius
            truth.source[src_idx].polar_pos = cart2pol(pol_pos)

            # Ground-truth VAD
            vad_vector = np.loadtxt(os.path.join(array_dir, f'VAD_{this_array}_{src_idx}.txt'), skiprows=1)
            len_audio = vad_vector.shape[0]
            t_stamps_audio = np.arange(0, len_audio).astype(np.float) / fs  # 48kHz
            VAD = np.zeros((truth.timestamps.shape[0]), dtype=np.int)  # 120Hz 
            cnt = 0
            for i in range(1, len_audio):
                if t_stamps_audio[i] >= truth.timestamps[cnt]:
                    VAD[cnt] = vad_vector[i - 1]
                    cnt += 1
                if cnt >= VAD.shape[0]:
                    break
            if cnt < VAD.shape[0]:
                VAD[cnt:] = vad_vector[-1]
                # Mismatch of up to 2 values to be expected
                if cnt < VAD.shape[0] - 2:
                    log.warning('VAD values do not match')

            # Calculate number of active periods
            vad_cnt = 0
            for i in range(1, VAD.shape[0]):
                if (VAD[i - 1] == 0) and (VAD[i] == 1):  # speech onset
                    vad_cnt += 1

            if np.sum(VAD) == 0:
                raise ValueError('No VAD values found!')

            # Store VAD ground-truth in structure
            truth.source[src_idx].VAD = Namespace(
                num_periods=np.amax([1, vad_cnt]),  # equal to one if no VAD is performed
                activity = VAD  # VAD decisions for each timestamp
            )

    return truth
