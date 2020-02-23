# -*- coding: utf-8 -*-
# Copyright 2019 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import Namespace
import h5py
import glob
import logging
import numpy as np
import pandas as pd
import os
import timeit

from locata_wrapper.utils.check import CheckResults
from locata_wrapper.utils.load_data import GetTruth
from locata_wrapper.utils.load_data import LoadData
from locata_wrapper.utils.metrics import Measures

from matplotlib import pyplot as plt


def ElapsedTime(time_array):
    n_steps = time_array.shape[0]
    elapsed_time = np.zeros([n_steps])
    for i in range(1, n_steps):
        elapsed_time[i] = (time_array[i] - time_array[i - 1]).total_seconds()
    return np.cumsum(elapsed_time)


class ProcessTask():
    def __init__(self, args, algorithm, opts, log=logging):
        self.data_dir = args.data_dir
        self.log = log
        self.is_dev = args.is_dev
        self.arrays = args.arrays
        self.algorithm = algorithm
        self.opts = opts
        self.results_dir = args.results_dir
        self._task_process = args.task_process
        self._save_results = args.save_results
        self._save_plots = args.save_plots
        pass
    
    def __call__(self, this_task):
        task_dir = os.path.join(self.data_dir, 'task{}'.format(this_task))

        # Read all recording IDs available for this task:
        recordings = sorted(glob.glob(os.path.join(task_dir, '*')))

        # Parse through all recordings within this task:
        for this_recording in recordings:
            recording_id = int(this_recording.split('recording')[1])

            # Read all recording IDs available for this task:
            array_names = glob.glob(os.path.join(this_recording, '*'))
            for array_dir in array_names:
                this_array = os.path.basename(array_dir)
                if this_array not in self.arrays:
                    continue
                self.log.info('Processing task {}, recording {}, array {}.'.format(this_task, recording_id, this_array))
                # Load data

                # Load data from csv / wav files in database:
                audio_array, _, position_array, position_source, required_time = LoadData(
                    array_dir, self.log, self.is_dev)
                
                # Extract ground truth

                # position_array stores all optitrack measurements.
                # Extract valid measurements only (specified by required_time.valid_flag).
                groundtruth = GetTruth(this_array, position_array, position_source, required_time, self.is_dev)

                self.log.debug('Processing Complete!')

                if 'Forward' in self._task_process:
                    results = self.forward(
                        this_array, audio_array, position_array, array_dir, required_time, groundtruth)
                else:
                    results = self.get_results(array_dir)
                
                if 'Eval' in self._task_process:
                    Measures(groundtruth, results)

    def get_results(self, array_dir):

        return results
    
    def forward(self, this_array, audio_array, position_array, array_dir, required_time, truth):
        # Load signal
        in_localization = Namespace()

        # Get number of mics and mic array geometry:
        in_localization.numMics = position_array.data[this_array].mic.shape[2]

        # Signal and sampling frequency:
        in_localization.y = audio_array.data[this_array]    # signal
        in_localization.fs = audio_array.fs                 # sampling freq

        # Users must provide estimates for each time stamp in in.timestamps

        # Time stamps required for evaluation
        in_localization.timestamps = ElapsedTime(required_time.time)[required_time.valid_flag]
        in_localization.time = required_time.time[required_time.valid_flag]
        in_localization.array = truth.array
        in_localization.array_name = this_array
        in_localization.mic_geom = truth.array.mic

        self.log.info('...Running localization using {}'.format(self.algorithm.__name__))
        # start_time = timeit.default_timer()
        results = self.algorithm(in_localization, self.opts)

        # results.telapsed = timeit.default_timer() - start_time

        # Check results structure is provided in correct format
        # CheckResults(results, in_localization, opts, log)
        # Plots & Save results to file

        self.log.info('Localization Complete!')
        if self._save_results:
            x_axis = np.average(audio_array.data[this_array], axis=1)
            x_len = x_axis.shape[0]
            fs = audio_array.fs 

            _idx = [x for x in truth.source]
            # Create directory for this array in results directory
            result_dir = array_dir.replace(self.data_dir, self.results_dir)
            os.makedirs(result_dir, exist_ok=True)

            for source_id in range(len(results.source)):
                df = pd.DataFrame(results.source[source_id])
                _source_id = _idx[source_id]
                filename = os.path.join(result_dir, 'source_{}'.format(source_id + 1))
                df.to_csv(f'{filename}.txt', index=False, sep='\t', encoding='utf-8')
                if self._save_plots:
                    self.save_plots()
        return results
    
    def save_plots(self):
        azu_x_pd = np.degrees(df[['azimuth']].values)
        #azu_x_gt = np.degrees(truth.source[_source_id].polar_pos[:, 0])
        azu_t = np.linspace(0, x_len / fs, azu_x_gt.shape[0])

        ele_x_pd = np.degrees(df[['elevation']].values)
        # ele_x_gt = np.degrees(truth.source[_source_id].polar_pos[:, 1])
        ele_t = np.linspace(0, x_len / fs, ele_x_gt.shape[0])

        fig, ax = plt.subplots(3, figsize=(4, 6))
        fig.tight_layout(pad=2.0)
        # plt.plot(azu_t, azu_x_gt, 'ob', azu_t, azu_x_pd, 'xr')
        ax[0].plot(t_axis, x_axis)
        ax[0].set_title(f'Task {this_task}, recording {recording_id}, array: {this_array}')
        ax[0].set(xlabel='Time, $t$, [s]', ylabel='Amplitude')

        ax[1].plot(azu_t, azu_x_gt, '.b', label='groundtruth')
        ax[1].plot(azu_t, azu_x_pd, 'xr', label='estimate')
        ax[1].set(xlabel='Time, $t$, [s]', ylabel='Azimuth [deg]')
        ax[1].legend()

        ax[2].plot(ele_t, ele_x_gt, '.b', label='groundtruth')
        ax[2].plot(ele_t, ele_x_pd, 'xr', label='estimate')
        ax[2].set(xlabel='Time, $t$, [s]', ylabel='Elevation [deg]')
        ax[2].legend()

        fig.savefig(f'{filename}.png')
