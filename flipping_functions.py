#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:04:48 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd
import tkinter as tk
from scipy.io import loadmat
from os.path import join
from iblutil.util import Bunch
from flipping_paths import DATA_PATH, FIG_PATH


def paths():
    paths = dict()
    paths['data_path'] = DATA_PATH
    paths['fig_path'] = FIG_PATH
    return paths


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                 "axes.titlesize": 7,
                 "axes.labelsize": 7,
                 "axes.linewidth": 0.5,
                 "lines.linewidth": 1,
                 "lines.markersize": 3,
                 "xtick.labelsize": 7,
                 "ytick.labelsize": 7,
                 "savefig.transparent": True,
                 "xtick.major.size": 2.5,
                 "ytick.major.size": 2.5,
                 "xtick.major.width": 0.5,
                 "ytick.major.width": 0.5,
                 "xtick.minor.size": 2,
                 "ytick.minor.size": 2,
                 "xtick.minor.width": 0.5,
                 "ytick.minor.width": 0.5,
                 'legend.fontsize': 7,
                 'legend.title_fontsize': 7
                 })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    colors = {'sert': sns.color_palette('Dark2')[0],
              'wt': [0.75, 0.75, 0.75],
              'stim': sns.color_palette('colorblind')[9],
              'no-stim': sns.color_palette('colorblind')[7]}
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


def load_session(ses_path):

    # Load in MAT file
    data = loadmat(join(ses_path, 'Behavior&Spikes.mat'))

    # Get event times
    events = Bunch()
    events.first_lick_times = np.squeeze(data['events'][0][0][0][0][2].T)
    events.opto_stimulation = np.squeeze(data['Stim'].T)
    if events.first_lick_times.shape[0] < events.opto_stimulation.shape[0]:
        events.opto_stimulation = events.opto_stimulation[:events.first_lick_times.shape[0]]


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


def get_subjects():
    subjects_df = pd.DataFrame(data={
        'subject': ['FC096', 'FC097', 'EA107', 'EA111', 'FC094', 'EA106'],
        'sert-cre': [1, 1, 1, 1, 0, 0]})
    return subjects_df
