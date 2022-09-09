#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:04:48 2022
By: Guido Meijer
"""

import numpy as np
from scipy.io import loadmat
from os.path import join
from iblutil.util import Bunch
from flipping_paths import DATA_PATH, FIG_PATH


def paths():
    paths = dict()
    paths['data_path'] = DATA_PATH
    paths['fig_path'] = FIG_PATH
    return paths


def load_session(ses_path):

    # Load in MAT file
    data = loadmat(join(ses_path, 'Behavior_Spikes.mat'))

    # Get event times
    events = Bunch()
    events.first_lick = data['events'][0][0][0][0][2].T
    events.trials = data['events'][0][0][0][0][0].T

    # Get spike times
    spikes = Bunch()
    spikes.times, spikes.clusters = [], []
    for i in range(data['spikes'].shape[0]):
        spikes.times.append(np.squeeze(data['spikes'][i][0].T))
        spikes.clusters.append([i] * np.squeeze(data['spikes'][i][0].T).shape[0])
    spikes.times = np.concatenate(np.array(spikes.times, dtype=object))
    spikes.clusters = np.concatenate(np.array(spikes.clusters, dtype=object))
    spike_order = np.argsort(spikes.times)
    spikes.times = spikes.times[spike_order]
    spikes.clusters = spikes.clusters[spike_order]

    # Clusters is empty for now
    clusters = Bunch()

    return spikes, clusters, events