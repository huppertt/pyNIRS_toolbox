import numpy as np
import pint
from pyBrainAnalyzIR.math.corrcoef_robust import corrcoef as robust_corrcoef

import pandas as pd
import xarray as xr
import itertools

from cedalion.math.ar_model import ar_filter


def compute_connectivity(data, robust=True, AR=None, stimMasks=None):

    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')

    data = data - data.mean("time")
    data = data - data.mean("time")

    if AR is not None and AR > 0:
        data = ar_filter(data, pmax=AR)
        # d.pint.dequalify()

    shp = data.shape
    values = data.data
    if isinstance(values, pint.Quantity):
        values = values.magnitude
    d2 = np.reshape(values, (shp[0], shp[1] * shp[2]))

    if stimMasks is None:
        stimName = ['rest']
        stimList = [np.arange(0, shp[0])]

    df_all = pd.DataFrame()

    for stimName, stimList in zip(stimName, stimList):

        if robust:
            r, p = robust_corrcoef(d2[stimList, :], verbose=False)
        else:
            r, p = np.corrcoef(d2[stimList, :], rowvar=False)

        nchan = shp[1]
        ntype = shp[2]

        if (hasattr(data, 'wavelength')):
            types = np.tile(data.wavelength, (nchan, 1)).T.flatten()
        else:
            types = np.tile(data.chromo, (nchan, 1)).T.flatten()

        chans = np.tile(data.channel, (ntype, 1)).flatten()

        chan_from = []
        chan_to = []
        type_to = []
        type_from = []
        coef = []
        pval = []

        for i in range(0, len(chans)):
            for j in range(0, len(chans)):
                chan_from.append(chans[i])
                chan_to.append(chans[j])
                type_from.append(types[i])
                type_to.append(types[j])
                coef.append(r[i, j])
                pval.append(p[i, j])

        df = pd.DataFrame({'Coeff': coef, 'Pvalue': pval, 'Condition': stimName,
                          'Channel_to': chan_to, 'Channel_from': chan_from,
                           'Type_to': type_to, 'Type_from': type_from})

        df_all = pd.concat([df_all, df], ignore_index=True)

    # Compute connectivity matrix from the input data
    return df_all


def compute_hyperscanning(datadictionary, robust=True, AR=None, stimMasks=None, unordered=False):

    subjects = list(datadictionary.keys())
    data_processed = []

    channel_info = pd.DataFrame()

    tstart = -np.inf
    tend = np.inf
    dt = 0
    for d in datadictionary.values():
        t1 = d.time.values
        tstart = np.max([tstart, t1[0]])
        tend = np.min([tend, t1[-1]])
        dt = np.max([dt, t1[2] - t1[1]])

    tall = np.arange(tstart, tend, dt)
    tall = xr.DataArray(tall, dims='time', coords={'time': tall})

    for subj in subjects:
        data = datadictionary[subj].copy()

        if (hasattr(data, 'wavelength')):
            data = data.transpose('time', 'channel', 'wavelength')
        else:
            data = data.transpose('time', 'channel', 'chromo')

        data = data - data.mean("time")
        data = data - data.mean("time")
        data = data.pint.interp(time=tall, method='linear')

        if AR is not None and AR > 0:
            data = ar_filter(data, pmax=AR)

        shp = data.shape
        nchan = shp[1]
        ntype = shp[2]

        values = data.data
        if isinstance(values, pint.Quantity):
            values = values.magnitude
        dB = np.reshape(values, (shp[0], shp[1] * shp[2]))
        if len(data_processed) > 0:
            data_processed = np.concatenate((data_processed, dB), axis=1)
        else:
            data_processed = dB

        if (hasattr(data, 'wavelength')):
            types = np.tile(data.wavelength, (nchan, 1)).T.flatten()
        else:
            types = np.tile(data.chromo, (nchan, 1)).T.flatten()

        chans = np.tile(data.channel, (ntype, 1)).flatten()

        channel_info = pd.concat([channel_info, pd.DataFrame({'Subject': subj,
                                                              'Channel': chans, 'Type': types})],
                                 ignore_index=True)

    if stimMasks is None:
        stimName = ['rest']
        stimList = [np.arange(0, len(tall))]

    if unordered:
        ntps = data_processed.shape[0]
        perm_iterator = itertools.permutations(subjects)
        # Convert the iterator to a list of lists
        all_permutations = [list(p) for p in perm_iterator]

        lst = {}
        for idx, sub in enumerate(subjects):
            lst[sub] = np.where(channel_info.Subject == sub)[0]

        data_processed_unordered = []
        stimList_unordered = []
        for idx, perm in enumerate(all_permutations):

            data_processed_local = np.concatenate([data_processed[:, lst[sub]] for sub in perm], axis=1)

            stimList_local = stimList.copy()
            stimList_local = stimList_local + idx * ntps * np.ones(len(stimList_local), dtype=int)

            if len(data_processed_unordered) == 0:
                data_processed_unordered = data_processed_local
                stimList_unordered = stimList_local
            else:
                data_processed_unordered = np.concatenate((data_processed_unordered, data_processed_local), axis=0)
                stimList_unordered = np.concatenate((stimList_unordered, stimList_local), axis=1)

        data_processed = data_processed_unordered
        stimList = stimList_unordered

    df_all = pd.DataFrame()

    for stimName, stimList in zip(stimName, stimList):
        if robust:
            r, p = robust_corrcoef(data_processed[stimList, :], verbose=False)
        else:
            r, p = np.corrcoef(data_processed[stimList, :], rowvar=False)

        chan_from = []
        chan_to = []
        type_to = []
        type_from = []
        subject_to = []
        subject_from = []
        coef = []
        pval = []

        for rowto in channel_info.itertuples():
            for rowfrom in channel_info.itertuples():
                chan_from.append(rowfrom.Channel)
                chan_to.append(rowto.Channel)
                type_from.append(rowfrom.Type)
                type_to.append(rowto.Type)
                subject_from.append(rowfrom.Subject)
                subject_to.append(rowto.Subject)
                coef.append(r[rowfrom.Index, rowto.Index])
                pval.append(p[rowfrom.Index, rowto.Index])

        df = pd.DataFrame({'Coeff': coef, 'Pvalue': pval, 'Condition': stimName,
                          'Channel_to': chan_to, 'Channel_from': chan_from,
                           'Type_to': type_to, 'Type_from': type_from,
                           'Subject_to': subject_to, 'Subject_from': subject_from})

        df_all = pd.concat([df_all, df], ignore_index=True)

    # Compute connectivity matrix from the input data
    return df_all
