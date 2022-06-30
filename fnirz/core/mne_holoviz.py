
from collections import OrderedDict
import itertools
import re
import pandas as pd

import mne
from mne.preprocessing.nirs import optical_density

import holoviews as hv
from holoviews.operation.datashader import datashade
import datashader as ds
from bokeh.models import HoverTool
hv.extension('bokeh')

from fnirz.util import melt_mne_OD

class MNEdash:
    def __init__(self, raw_OD) -> None:
        self.raw_OD = raw_OD
        self.df_offset_melt, self.chYticks = melt_mne_OD(self.raw_OD)
        self.ch_wl_fnirs_curves = self.create_ch_curves()
        self.ch_tooltip = self.chan_tooltip()
        self.ch_Nd = self.plot_chans()
        self.ch_list = list(self.ch_Nd.keys())

    def get_chan_data(self):
        return self.raw_OD.info['chs']

    def get_chan_loc(self):
        # collect channel locations into list of dicts
        chs = self.get_chan_data()
        ch_loc = []
        for ci, ch in enumerate(chs):
            ich_dict = {}
            ich_dict['channel'] = ch['ch_name'].split(' ')[0]
            ich_dict['sourceX'] = ch['loc'][3]
            ich_dict['sourceY'] = ch['loc'][4]
            ich_dict['detectorX'] = ch['loc'][6]
            ich_dict['detectorY'] = ch['loc'][7]
            ch_loc.append(ich_dict)
        return ch_loc

    def create_hv_chan_dict(self):
        # create orderedDict of hv Curves from channels
        ch_loc = self.get_chan_loc()
        ch_loc_curves = OrderedDict()
        for ch in ch_loc:
            ch_loc_curves[ch['channel']] = hv.Curve([(ch['sourceX'], ch['sourceY']),
                                                (ch['detectorX'], ch['detectorY'])])
        return ch_loc_curves

    def chan_tooltip(self):
        # create custom hover tooltip for the channels
        tooltips = [('channel', '@channel')]
        hover = HoverTool(tooltips=tooltips)
        return hover

    def plot_chans(self):
        # overlay hv curves of channels 
        ch_loc_curves = self.create_hv_chan_dict()
        ch_Nd = hv.NdOverlay(ch_loc_curves, kdims=['channel']).opts({
            'Curve': {'line_width':5, 'alpha':.5, 'color':'black'},
            'NdOverlay': {'tools':['tap', self.ch_tooltip, 'box_select'],
                        'xaxis':None, 'yaxis':None,
                        'title':'Relative Channel Location',
                        'show_legend':False, 'width':400, 'height':400}})
        return ch_Nd

    def chan_sel_stream(self, ch_Nd):
        # create selection stream for channels
        return hv.streams.Selection1D(source=ch_Nd)

    def plot_optodes(self):
        # make optode plot
        chs = self.get_chan_data()
        optodeDict = {}

        for ci, ch in enumerate(chs):
            sourceStr = ch['ch_name'].split('_')[0]
            sourceInt = [int(s) for s in re.findall(r'\d+', sourceStr)][0]
            sourcePos = ch['loc'][3:6]
            sourceX = ch['loc'][3]
            sourceY = ch['loc'][4]
            optodeDict[sourceStr] = {'id': sourceStr, 'type': 'source', 'typeNum': sourceInt, \
                'pos': sourcePos, 'x': sourceX, 'y':sourceY}
            
            detectorStr = ch['ch_name'].split('_')[1].split(' ')[0]
            detectorInt = [int(s) for s in re.findall(r'\d+', detectorStr)][0]
            detectorPos = ch['loc'][6:9]
            detectorX = ch['loc'][6]
            detectorY = ch['loc'][7]
            optodeDict[detectorStr] = {'id': detectorStr, 'type': 'detector', \
                'typeNum': detectorInt, 'pos': detectorPos, 'x':detectorX, 'y':detectorY}

        optodeDF = pd.DataFrame.from_dict(optodeDict).T  
        optodePoints = hv.Points(optodeDF, kdims=['x','y'], vdims=['id', 'type']).opts(
            color='type', size=25, cmap=['#ff0000', '#0000ff'], alpha=1, tools=['hover'],
            show_legend=False) #width=500,
        optodeLabels = hv.Labels(optodeDF, kdims=['x','y'],vdims=['id', 'type']).opts(
            text_color='white', text_align='center')
        optodes_plot = optodePoints * optodeLabels

        return optodes_plot

    def plot_triggers(self):
        # Make triggers plot
        triggersOnset = self.raw_OD.annotations.onset
        triggersDuration = self.raw_OD.annotations.duration
        triggers = pd.DataFrame.from_dict({'id':self.raw_OD.annotations.description.astype('int'), 'start':triggersOnset, 'duration':triggersDuration})
        condition = {1:'Right', 2:'Left'}
        triggers['type'] = triggers.id.map(condition)
        triggers['duration'] = 10
        triggers['end'] = triggers['start'] + triggers['duration']
        trigDict = {}
        colors = ['#BFB9E2', '#A8D0C6'] # ['#dadada', '#a9a9a9'] 
        for i,t in triggers.iterrows():
            trigDict[i] = hv.VSpan(t.start,t.end).opts(color=colors[t.id-1], alpha=.4)
        return hv.HoloMap(trigDict).overlay()


    def create_ch_curves(self):
        ch_wl_fnirs_curves = {i: hv.Curve(c, 'time', 'amplitude') \
            for i, c in self.df_offset_melt.groupby(['channel', 'wavelength'])}
        return ch_wl_fnirs_curves

    def get_ch_curves(self, channel, wavelength):
        # get all curves of (channel, wavelength)
        if not isinstance(channel, list):
            channel = [channel]
        if not isinstance(wavelength, list):
            wavelength = [wavelength]
        sel_fnirs_curves = {i: self.ch_wl_fnirs_curves[i] 
                        for i in list(itertools.product(channel, wavelength))}
        return sel_fnirs_curves

    def plot_fnirs_dmap(self, sel_fnirs_curves,  alpha=1):
        # return hv NdOverlay of fNIRS curves
        fnirs_Nd = hv.NdOverlay(sel_fnirs_curves, ['channel', 'wavelength']).opts({
            'NdOverlay': {'tools':['tap', self.ch_tooltip, 'box_select'],'show_legend':False,
                        'width':600, 'height':500, 'padding':0, 'framewise':True,
                        'yticks':self.chYticks, 'fontsize':{'yticks':8}, 'ylabel':'channel', 
                        'xlabel':'time (s)', 'title':'fNIRS - Optical Density'},
            'Curve': {'line_width':.5, 'color':'black', 'alpha':alpha}})
        return fnirs_Nd

    def make_fnirs_overlay(self, index, wavelength=760):
        # select and return hv DynamicMap of fNIRS data
        if not index:
            channel = self.ch_list
        else:
            channel = [self.ch_list[i] for i in index]
        fnirs_nd = self.plot_fnirs_dmap(self.get_ch_curves(channel, wavelength))
        return  fnirs_nd

    def plot(self):
        # create dmap linked to channel selection and constrain wavelength param
        optodes = self.plot_optodes()
        trigHM = self.plot_triggers()
        ch_stream = self.chan_sel_stream(self.ch_Nd)
        fnirs_dmap = hv.DynamicMap(self.make_fnirs_overlay, kdims=['wavelength'],
            streams=[ch_stream]).opts(framewise=True)
        fnirs_dmap = fnirs_dmap.redim.values(wavelength=[760, 850])

        return trigHM * fnirs_dmap + self.ch_Nd * optodes



