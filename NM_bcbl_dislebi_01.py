#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 13:57:39 2023
@author: Amit Jaiswal <amit.jaiswal@megin.fi>
This script was written for an collaborative project wih BCBL, Donostia, Spain.
"""
import mne
from os import cpu_count
from os.path import join
import numpy as np
from pathlib import Path 
script_path = Path( __file__ ).absolute() 
import os
os.chdir(os.path.dirname(script_path))
from utils.meginpy.process import my_var_cut_fn
n_jobs = cpu_count()
# from matplotlib.pyplot import close

# set parameters
pars = dict(bpfreq   = [2., 47.],
           linefreq  = 50.,
           stimchan  = 'STI101',
           more_plots= True,
           varcut    = [2 ,97])
print(pars)

#%% Read data and preprocess.
if os.uname()[1] in ['dell7770', 'palmu2']:
    subjects_dir, meg_dir = 'subjects_dir', '/net/qnap/data/rd/ChildBrain/DATA/bcb_dyslex/'
else:
    subjects_dir, meg_dir = 'subjects_dir', 'put your path'

# Read raw data and events
subject = 'sub001'
fname = join(meg_dir, '02_dislebi_c_dlr_semantic_raw_tsss.fif')
raw   = mne.io.read_raw_fif(fname, allow_maxshield=True, preload=False, verbose=True)
raw.load_data(verbose=True)
print(np.int32(np.unique(raw._data[raw.ch_names.index('STI101')])))
events = mne.find_events(raw, stim_channel=pars['stimchan'], min_duration=0.002, shortest_event=1)
eventIDs = np.unique(events[:,2])
mne.viz.plot_events(events, sfreq=raw.info['sfreq']) if pars['more_plots'] else None

#%% Plot to check, channels, select channels and segment data
raw.plot(events)                        if pars['more_plots'] else None
raw.plot_psd(fmax=pars['bpfreq'][1]+10) if pars['more_plots'] else None

picks_bio = mne.pick_types(raw.info, meg=True, ecg=True, eog=True, eeg=True)
picks_meg = mne.pick_types(raw.info, meg=True)
picks_mag = mne.pick_types(raw.info, meg='mag')
picks_grad= mne.pick_types(raw.info, meg='grad')

if pars['bpfreq'][1]>pars['linefreq']-2:
    raw.notch_filter(pars['linefreq'], picks=picks_bio, filter_length='auto', 
                     notch_widths=None, trans_bandwidth=1.0, n_jobs=n_jobs, method='fir')
raw.filter(pars['bpfreq'][0], pars['bpfreq'][1], picks=None, filter_length='auto', 
           l_trans_bandwidth=1.0, h_trans_bandwidth=1.0, n_jobs=n_jobs, method='fir')
raw.plot(events)                        if pars['more_plots'] else None
raw.plot_psd(fmax=pars['bpfreq'][1]+10) if pars['more_plots'] else None

#%% Apply ICA



#%% Make trials 
epochs, evokeds = {}, {}
for ii in eventIDs:
    epochs[ii] = mne.Epochs(raw, events, event_id=ii, 
                    tmin=-0.5, tmax=0.5,  baseline=None,  preload=True)
    epochs[ii].pick_types(meg=True)
    bad_trials = my_var_cut_fn(epochs[ii], pars['varcut'][0], pars['varcut'][1], 
                                mode=1, to_plot=False, predef_bads=[])
    epochs[ii].drop(bad_trials, reason=f"Variance-based rejection {pars['varcut']}", verbose=None)
    evokeds[ii] = epochs[ii].average()
    # evokeds[ii].comment = str(ii)
    # evokeds[ii].plot(spatial_colors=True, window_title=f'{evokeds[ii].comment[0]}')
mne.viz.plot_evoked_topo(list(evokeds.values())[:-1])

#%% 



