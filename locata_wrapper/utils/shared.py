# -*- coding: utf-8 -*-
# Copyright 2020 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Code transcribed from MATLAB

import numpy as np


def wrapTo2Pi(_lambda):
    """Wrap angle in radians to [0 pi]"""
    positiveInput = _lambda > 0
    _lambda = np.mod(_lambda, 2 * np.pi)
    _idx = (_lambda == 0) * positiveInput
    _lambda[_idx] = 2 * np.pi
    return _lambda


def wrapToPi(_lambda):
    """Wrap angle in radians to [-pi pi]"""
    q = (_lambda < -np.pi) + (np.pi < _lambda)
    _lambda[q] = wrapTo2Pi(_lambda[q] + np.pi) - np.pi
    return _lambda
