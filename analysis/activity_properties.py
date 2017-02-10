#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tools to compute the properties of the activity """

import numpy as np
import scipy.sparse as ssp


# ------------------- #
# Get data from NEST  #
# ------------------- #

def get_data(recorders):
    ''' Return spike and variables' data '''
    import nest
    # spikes
    data = nest.GetStatus(recorders[0])[0]["events"]
    spike_times = data["times"]
    senders = data["senders"]
    # variables
    if len(recorders) > 1:
        base_data2 = nest.GetStatus(recorders[1])
        data2 = [d["events"] for d in base_data2]
        time_var = np.array(data2[0]["times"])
        data_var = {key: [] for key in data2[0]
                    if key not in ("senders", "times")}
        for d in data2:
            for key, val in d.items():
                if key not in ("senders", "times"):
                    data_var[key].append(val)
    return spike_times, senders, time_var, data_var


def get_spikes(recorder=None, spike_times=None, senders=None):
    '''
    Return a 2D sparse matrix, where:
        - each row i contains the spikes of neuron i
        - each column j contains the times of the jth spike for all neurons

    Parameters
    ----------
    recorder : tuple, optional (default: None)
        Tuple of NEST gids, where the first one should point to the
        spike_detector which recorded the spikes.
    spike_times : array-like, optional (default: None)
        If `recorder` is not provided, the spikes' data can be passed directly
        through their `spike_times` and the associated `senders`.
    senders : array-like, optional (default: None)
        `senders[i]` corresponds to the neuron which fired at `spike_times[i]`.

    Example
    -------
    >> get_spikes()

    >> get_spikes(recorder)

    >> times = [1.5, 2.68, 125.6]
    >> neuron_ids = [12, 0, 65]
    >> get_spikes(spike_times=times, senders=neuron_ids)

    Note
    ----
    This function supposes that neuron GIDs for a continuous set of integers.
    If no arguments are passed to the function, the first spike_recorder
    available in NEST will be used.

    Returns
    -------
    CSR matrix containing the spikes sorted by neuron (rows) and time
    (columns).
    '''
    import nest
    # get spikes
    if recorder is not None:
        data = nest.GetStatus(recorder[0])[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]
        senders -= np.min(senders)
    elif spike_times is None and senders is None:
        nodes = nest.GetNodes(
            (0,), properties={'model': 'spike_detector'})
        data = nest.GetStatus(nodes[0])[0]["events"]
        spike_times = data["times"]
        senders = data["senders"]
        senders -= np.min(senders)
    # create the sparse matrix
    data = [0 for _ in range(len(set(senders)))]
    row_idx = []
    col_idx = []
    for time, neuron in zip(spike_times, senders):
        row_idx.append(neuron)
        col_idx.append(data[neuron])
        data[neuron] += 1
    return ssp.csr_matrix((spike_times, (row_idx, col_idx)))


# ------------------------------------- #
# Analyze properties from spike trains  #
# ------------------------------------- #


def interburst_properties(bursts, current_index, steady_state, times, variables,
                          resolution, result):
    '''
    Find the end of the previous burst, then compute the interburst (IBI)
    duration and the extremal values of the variables during the interburst.
    '''
    current_burst = bursts[current_index + steady_state]
    # Time
    IBI_start = bursts[current_index + steady_state-1][1]
    IBI_end = current_burst[0]
    result["IBI"] += IBI_end - IBI_start
    # time slice of the IBI to array indices
    idx_start = np.argwhere(times >= IBI_start)[0][0]
    idx_end = np.argwhere(times < IBI_end-resolution)[-1][0]
    idx_wmax = np.argwhere(times > current_burst[1])[0][0]
    for varname, varvalues in iter(variables.items()):
        varname = "V" if varname == "V_m" else varname
        result[varname + "_min"] += np.min(varvalues[0][idx_start:idx_end])
        result[varname + "_max"] += np.max(varvalues[0][idx_start:idx_end])


def spiking_properties(burst, spike_times, senders, result):
    '''
    Compute the average and standard deviation of the interspike interval (ISI)
    as well as those of the number of spikes during the burst.
    '''
    # get the spikes inside the burst
    spikes = np.where( (spike_times >= burst[0])*(spike_times <= burst[1]) )[0]
    # get the number and ISI for each spike, then average
    lst_num_spikes = []
    lst_ISI = []
    for sender in set(senders[spikes]):
        subset = np.where(senders[spikes] == sender)[0]
        lst_num_spikes.append(len(subset))
        stimes = spike_times[spikes][subset]
        if len(stimes) > 1:
            lst_ISI.extend(np.diff(stimes))
    result["num_spikes"] += np.average(lst_num_spikes)
    result["std_num_spikes"] += np.std(lst_num_spikes)
    result["ISI"] += np.average(lst_ISI)
    result["std_ISI"] += np.std(lst_ISI)

