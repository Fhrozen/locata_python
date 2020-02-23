#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 Waseda University (Nelson Yalta)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import Namespace
import locata_wrapper.utils as locata_utils
import logging

import os
from sacred import Experiment
# from sacred.observers import MongoObserver

import sys


ex = Experiment()
logging.basicConfig(format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
logger = logging.getLogger('my_custom_logger')
ex.logger = logger

# ex.observers.append(MongoObserver.create())


@ex.config
def config_eval():
    """Inputs:

    data_dir:    String with directory path for the LOCATA/DCASE Dev or Eval database
    results_dir: String with directory path in which to save the results of this
                 function
    is_dev:      Kind of database specified by data_dir
                 False: Eval database
                 True: Dev database
    arrays:      List with array names which should be evaluated (optional)
                 LOCATA list {'benchmark2', 'eigenmike', 'dicit','dummy'} is taken
                 as default which contains all available arrays
                 DCASE list {'doa', 'foa'}
    tasks:       List with task(s) (optional)
                 LOCATA List [1,2,3,4,5,6] is taken as default which evaluates
                 over all available tasks
                 DCASE list [1 2 3 4]

    Outputs: N/A (saves results as csv files in save_dir)
    """
    # [locata or dcase]
    data_dir = './data'  # NOQA
    results_dir = './results'  # NOQA
    is_dev = True  # NOQA
    arrays = ['benchmark2', 'eigenmike', 'dicit', 'dummy']  # NOQA
    tasks = [1, 2, 3, 4, 5, 6]  # NOQA
    algorithm = 'locata_wrapper.algorithm.music:MUSIC'  # NOQA
    processes = 1  # NOQA
    task_process = 'ForwardAndEval'  # NOQA
    save_results = True  # NOQA


@ex.main
def main_eval(_config, _log):
    # Straight copy or use of prt for _config generates error due to ReadOnlyList/Dict
    # to avoid problems for multiprocessing each value is copy and in for ReadOnlyList
    # these are converter to lists
    args = Namespace()
    for _value in [x for x in _config if '__' not in x]:
        if 'List' in str(type(_config[_value])):
            setattr(args, _value, list(_config[_value]))
        else:
            setattr(args, _value, _config[_value])

    # Selection of the localisation algorithm

    # Enter the name of the PYTHON function of your localization algorithm.
    # The LOCATA organizers provided MUSIC here as an example for the required interface.
    # Check the documentation inside for contents of structures.
    my_alg_name = locata_utils.DynamicImport(args.algorithm, log=_log)

    # Check and process input arguments

    # Create directories if they do not exist already
    if not os.path.exists(args.results_dir):
        _log.warning('Directory for results not found. New directory created.')
        os.makedirs(args.results_dir)

    if not os.path.exists(args.data_dir):
        _log.error('Incorrect data path!')
        sys.exit(1)

    # Initialize settings required for these scripts:
    opts = locata_utils.InitalOptions()

    # Check and process input arguments
    # check if input contains valid tasks
    error_tasks = [x for x in list(set(args.tasks)) if not 0 < x < 7]
    if len(error_tasks) > 1:
        _log.error('Invalid input for task number(s)')
        sys.exit(1)

    # check if input contains valid arrays names
    if args.arrays is None:
        args.arrays = opts.valid_arrays
    else:
        error_arrays = [x for x in list(set(args.arrays))
                        if x not in opts.valid_arrays]
    if len(error_arrays) > 1:
        _log.error('Invalid input for array(s)')
        sys.exit(1)

    _log.info('Available tasks in the dev dataset in {}: {}'.format(
        args.data_dir, args.tasks))

    # Process
    # Parse through all specified task folders
    # Task: Forward, Eval, and Forward and Eval
    task_process = locata_utils.ProcessTask(args, my_alg_name, opts, _log)

    # Multiprocessing
    if args.processes > 1:
        from pathos.multiprocessing import ProcessingPool
        pool = ProcessingPool(args.processes)

    for this_task in args.tasks:
        if args.processes > 1:
            pool.map(task_process, [this_task])
        else:
            task_process(this_task)
    if args.processes > 1:
        pool.close()
        pool.join()
    _log.info('Processing finished!')


if __name__ == '__main__':
    ex.run_commandline()
