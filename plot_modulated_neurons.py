#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:11:28 2022
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from flipping_functions import paths, figure_style
path_dict = paths()

# Load in dataframe
neurons_df = pd.read_csv(join(path_dict['data_path'], 'neurons_df.csv'))
sert_neurons = neurons_df[neurons_df['sert-cre'] == 1]
wt_neurons = neurons_df[neurons_df['sert-cre'] == 0]

# Calculate percentage modulated neurons
all_mice = (sert_neurons.groupby('subject').sum()['sig_mod'] / sert_neurons.groupby('subject').size() * 100).to_frame().reset_index()
all_mice['sert-cre'] = 1
wt_mice = (wt_neurons.groupby('subject').sum()['sig_mod'] / wt_neurons.groupby('subject').size() * 100).to_frame().reset_index()
wt_mice['sert-cre'] = 0
all_mice = pd.concat((all_mice, wt_mice), ignore_index=True)
all_mice = all_mice.rename({0: 'perc_mod'}, axis=1)

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1), dpi=dpi)

f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
#sns.stripplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=3,
#              palette=[colors['sert'], colors['wt']], ax=ax1, jitter=0.2)

sns.swarmplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=2.5,
              palette=[colors['sert'], colors['wt']], ax=ax1)
ax1.set(xticklabels=['SERT', 'WT'], ylabel='Mod. neurons (%)', ylim=[-1, 40], xlabel='',
        yticks=[0, 10, 20, 30, 40])

sns.despine(trim=True)
#plt.tight_layout()

plt.savefig(join(path_dict['fig_path'], 'light_mod_summary.jpg'), dpi=600)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

ax1.hist(sert_neurons.loc[sert_neurons['sig_mod'] == 0, 'mod_index'], color=colors['no-stim'], bins=6,
         label='Not sig.')
ax1.hist(sert_neurons.loc[sert_neurons['sig_mod'] == 1, 'mod_index'], color=colors['stim'], bins=25,
         label='Sig.')
ax1.set(xticks=[-1, -0.5, 0, 0.5, 1], yscale='log', ylim=[1, 1000], yticklabels=[1, 10, 100, 1000],
        ylabel='Neuron count', xlabel='Modulation index', title='SERT')
ax1.legend(frameon=False, prop={'size': 5}, loc='upper left')
sns.despine(trim=True)
plt.tight_layout()

ax2.hist(wt_neurons.loc[wt_neurons['sig_mod'] == 0, 'mod_index'], color=colors['no-stim'], bins=9,
         label='Not sig.')
ax2.hist(wt_neurons.loc[wt_neurons['sig_mod'] == 1, 'mod_index'], color=colors['stim'], bins=20,
         label='Sig.')
ax2.set(xticks=[-1, -0.5, 0, 0.5, 1], yscale='log', ylim=[1, 1000], yticklabels=[1, 10, 100, 1000],
        ylabel='Neuron count', xlabel='Modulation index', title='WT')
ax2.legend(frameon=False, prop={'size': 5}, loc='upper left')
sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(path_dict['fig_path'], 'modulation_index.jpg'), dpi=600)
