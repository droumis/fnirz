

import numpy as np
import pandas as pd

OFFSET_SCALE = .1

def melt_mne_OD(ODdata):
    data, times = ODdata.get_data(return_times=True) # data is (n_channels, n_times)
    df = pd.DataFrame(data=data.T, index=times, columns=ODdata.ch_names).rename_axis('time')

    # seperate the channel and wavelength column levels
    ch_wl = [(ch.split(' ')[0], int(ch.split('_')[1].split(' ')[1])) for ch in df.columns.values]
    cindex = pd.MultiIndex.from_tuples(ch_wl, names=["channel", "wavelength"])
    df.columns = cindex

    # create offset array to nicely stack timeseries traces in same plot
    offsetL = []
    chYticks = []
    for i, d in enumerate(df.groupby(axis=1, level='channel')):
        i = i*OFFSET_SCALE
        firstVal = d[1].values[0,:]
        offsetL.append(i - firstVal)
        chYticks.append((i, d[0]))
    offsetAr = np.concatenate(offsetL, axis=0)
    df_offset = df.add(offsetAr)
    
    # move the wavelength to a multiindex with time, 
    # and then reset the index so that we have a column for time and wavelength to melt
    df_offset_ready2melt = df_offset.stack(level='wavelength').reset_index()

    df_offset_melt = pd.melt(df_offset_ready2melt, id_vars=['time', 'wavelength'],
     var_name='channel', value_name='amplitude')

    return df_offset_melt, chYticks