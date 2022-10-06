#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:40:17 2022
By: Guido Meijer
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join, isdir, isfile
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from brainbox.task.closed_loop import (responsive_units, roc_single_event, differentiate_units,
                                       roc_between_two_events)
from flipping_functions import (paths, load_session, figure_style, get_subjects,
                                peri_multiple_events_time_histogram)
from sklearn.utils import shuffle
np.random.seed(42)  # fix random seed

# Settings
OVERWRITE = False
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

# Get df
if OVERWRITE:
    neurons_df = pd.DataFrame()
else:
    neurons_df = pd.read_csv(join(path_dict['save_path'], 'neurons_df.csv'))

# Loop over sessions
for i, session in enumerate(ses):

    # Get details
    subject = session[:5]
    ses_name = session[6:8]

    # Skip if data not there
    if ((subject not in subjects['subject'].values)
            or (isfile(join(path_dict['data_path'], session, 'Behavior&Spikes.mat')) == False)):
        print('Not found')
        continue

    # Skip if already done
    if ~OVERWRITE and ((subject in neurons_df['subject'].values) & (ses_name in neurons_df['session'].values)):
        print('Already done, skipping')
        continue

    # Get genotype
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]

    # Load in data
    print(f'Starting session {session}')
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
    neurons_df.to_csv(join(path_dict['save_path'], 'neurons_df.csv'))

    # Plot significant neurons
    if PLOT:
        colors, dpi = figure_style()
        for n, neuron_id in enumerate(neuron_ids[sig_mod]):
            # Plot PSTH
            p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            peri_multiple_events_time_histogram(
                spikes.times, spikes.clusters, events.first_lick_times, events.opto_stimulation,
                neuron_id, t_before=PLOT_PRE_TIME, t_after=PLOT_POST_TIME, bin_size=BIN_SIZE, ax=ax,
                pethline_kwargs=[{'color': colors['no-stim'], 'lw': 1}, {'color': colors['stim'], 'lw': 1}],
                errbar_kwargs=[{'color': colors['no-stim'], 'alpha': 0.3}, {'color': colors['stim'], 'alpha': 0.3}],
                raster_kwargs=[{'color': colors['no-stim'], 'lw': 0.5}, {'color': colors['stim'], 'lw': 0.5}],
                eventline_kwargs={'lw': 0}, include_raster=True)
            ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from first lick (s)',
                   yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3), xticks=[-1, 0, 1, 2, 3, 4])
            if np.round(ax.get_ylim()[1]) % 2 == 0:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            sns.despine(trim=False)
            plt.tight_layout()
            plt.savefig(join(fig_path, f'{subject}_{session}_neuron{neuron_id}.jpg'), dpi=600)
            plt.close(p)

# Save results
neurons_df.to_csv(join(path_dict['save_path'], 'neurons_df.csv'))