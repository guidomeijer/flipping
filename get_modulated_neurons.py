#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:40:17 2022
By: Guido Meijer
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join, isdir
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from brainbox.plot import peri_event_time_histogram
from brainbox.task.closed_loop import (responsive_units, roc_single_event, differentiate_units,
                                       roc_between_two_events)
from flipping_functions import paths, load_session, figure_style, get_subjects
from sklearn.utils import shuffle

# Settings
PRE_TIME = 1
POST_TIME = 1
PLOT_PRE_TIME = 1
PLOT_POST_TIME = 4
N_SHUFFLES = 500
BIN_SIZE = 0.05
PLOT = True

# Set paths
path_dict = paths()
fig_path = join(path_dict['fig_path'], 'SingleNeurons')

# Get subjects
subjects = get_subjects()

# Get sessions
ses = [name for name in os.listdir(path_dict['data_path']) if os.path.isdir(join(path_dict['data_path'], name))]

# Loop over sessions
neurons_df = pd.DataFrame()
for i, session in enumerate(ses):

    # Get details
    subject = session[:5]
    ses_name = session[6:8]
    if subject not in subjects['subject'].values:
        continue
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting session {session} ({i+1} of {len(ses)})')

    # Load in data
    spikes, clusters, events = load_session(join(path_dict['data_path'], session))

    # Get modulation index
    mod_index, neuron_ids = roc_between_two_events(spikes.times, spikes.clusters, events.first_lick_times,
                                                   events.opto_stimulation, post_time=POST_TIME)
    mod_index = 2 * (mod_index - 0.5)

    # Get null distribution of modulation index
    null_roc = np.empty((N_SHUFFLES, mod_index.shape[0]))
    for k in range(N_SHUFFLES):
        this_null_roc, _ = roc_between_two_events(spikes.times, spikes.clusters,
                                                  events.first_lick_times, shuffle(events.opto_stimulation),
                                                  post_time=POST_TIME)
        null_roc[k, :] = 2 * (this_null_roc - 0.5)

    # Get significant neurons
    sig_mod = ((mod_index > np.percentile(null_roc, 97.5, axis=0))
               | (mod_index < np.percentile(null_roc, 2.5, axis=0)))
    print(f'Found {(np.sum(sig_mod)/sig_mod.shape[0])*100:.1f}% modulated neurons')

    # Add to dataframe
    neurons_df = pd.concat((neurons_df, pd.DataFrame(data={
        'subject': subject, 'session': ses_name, 'sert-cre': sert_cre,
        'neuron_id': neuron_ids, 'mod_index': mod_index, 'sig_mod': sig_mod})))
    neurons_df.to_csv(join(path_dict['data_path'], 'neurons_df.csv'))

    # Plot significant neurons
    if PLOT:
        colors, dpi = figure_style()
        for n, neuron_id in enumerate(neuron_ids[sig_mod]):

            # Plot PSTH
            colors, dpi = figure_style()
            p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            peri_event_time_histogram(spikes.times, spikes.clusters,
                                      events.first_lick_times[events.opto_stimulation == 1],
                                      neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                      bin_size=BIN_SIZE,
                                      include_raster=False, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': colors['stim'], 'lw': 1},
                                      errbar_kwargs={'color': colors['stim'], 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            this_y_lim = ax.get_ylim()
            peri_event_time_histogram(spikes.times, spikes.clusters,
                                      events.first_lick_times[events.opto_stimulation == 0],
                                      neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME,
                                      bin_size=BIN_SIZE,
                                      include_raster=False, error_bars='sem', ax=ax,
                                      pethline_kwargs={'color': colors['no-stim'], 'lw': 1},
                                      errbar_kwargs={'color': colors['no-stim'], 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            ax.set(ylim=[np.min([this_y_lim[0], ax.get_ylim()[0]]),
                         np.max([this_y_lim[1], ax.get_ylim()[1]]) + np.max([this_y_lim[1], ax.get_ylim()[1]]) * 0.2])
            ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from first lick (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3), xticks=[-1, 0, 1, 2, 3, 4])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            sns.despine(trim=True, offset=2)
            plt.tight_layout()
            plt.savefig(join(fig_path, f'{subject}_{session}_neuron{neuron_id}.pdf'))
            plt.close(p)