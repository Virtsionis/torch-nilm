from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from nilmtk import DataSet


class MyChunk(Dataset):
    """MyChunk dataset."""

    def __init__(self, path, building, device,  dates=None, transform=None, window_size=50, test=False, mmax=None,**load_kwargs):
        """
        Args:
            path (string): Path to the h5 file.
            building(int): the desired building
            device(string): the desired device
            dates(list): list with the start and end(optional) dates for training window [start, end]
                        eg:['2016-04-01','2017-04-01']
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if test:
            self.mmax = mmax
        else:
            self.mmax = None
        self.path = path
        self.building = building
        self.device = device
        self.transform = transform

        #loads h5 file with Nilmtk DataSet class
        self.dataset = DataSet(path)

        self.window_size = window_size

        #use start/end date
        if(dates):
            self.dates = dates
            if(len(self.dates)>1):
                self.dataset.set_window(start=self.dates[0],end=self.dates[1])
            else:
                self.dataset.set_window(start=self.dates[0])

        #assign building
        elec1 = self.dataset.buildings[self.building].elec

        #mains : a nilmtk.ElecMeter object for the aggregate data
        #meter : a nilmtk.ElecMeter object for the meter data
        mains = elec1.mains()
        meter = elec1.submeters()[self.device]

        #get meter metadata
        self.meter_metadata = meter

        #these are nilmtk generators
        self.main_power_series = mains.power_series(**load_kwargs) 
        self.meter_power_series = meter.power_series(**load_kwargs)

        #get chunks
        mainchunk = next(self.main_power_series)
        meterchunk = next(self.meter_power_series)

        #get chunk details
        self.timeframe = mainchunk.timeframe
        self.chunk_name = mainchunk.name

        mainchunk = mainchunk[~mainchunk.index.duplicated()]

        #normalize chunks
        if self.mmax == None:
            self.mmax = mainchunk.max()
        mainchunk = mainchunk / self.mmax
        meterchunk = meterchunk / self.mmax

        #Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)

        #intersect chunks in order to have same length
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = np.array(mainchunk[ix])
        meterchunk = np.array(meterchunk[ix])

        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk)-self.window_size+1)[:, None]
        mainchunk = mainchunk[indexer]
        meterchunk = meterchunk[self.window_size-1:]

        #reshape and store to self attributes
        # mainchunk = np.reshape(mainchunk, (mainchunk.shape[0],1,1))
        # mainchunk = np.reshape(mainchunk, (mainchunk.shape[0],1))
        self.mainchunk = mainchunk

        # meterchunk = np.reshape(meterchunk, (len(meterchunk),-1 ))
        # meterchunk = np.reshape(meterchunk, (len(meterchunk), 1 ))
        self.meterchunk = meterchunk

    def __len__(self):
        return len(self.mainchunk)

    def __getitem__(self, i):
        x = torch.from_numpy(self.mainchunk)
        y = torch.from_numpy(self.meterchunk)
        return x[i].float(), y[i].float()

    def __mmax__(self):
        return self.mmax


class MyChunkList(Dataset):
    """MyChunkList dataset."""

    def __init__(self, buildings, device, filename=None, transform=None, **load_kwargs):
        """
        Args:
            filename (string): name of the train info file.
            buildings(list): list with the desired buildings
            device(string): the desired device
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.mmax = None
        self.buildings = buildings
        self.device = device
        self.transform = transform

        meterlist = []
        mainlist = []

        if filename == None:
            file = open('baseTrainSetsInfo_' + device, 'r')
        else:
            file = open(filename, 'r')
        
        for line in file:
            #take tokens from file
            toks = line.split(',')
            
            #loads h5 file with Nilmtk DataSet class
            train = DataSet(toks[0])
            
            #sets time window of each house
            print(toks[2],'-',toks[3])
            
            train.set_window(start=toks[2], end=toks[3])
            #sets buildings 
            train_elec = train.buildings[int(toks[1])].elec           
            
            #mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
            #meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
            meterlist.append(train_elec.submeters()[self.device])
            mainlist.append(train_elec.mains())
        
        file.close()
       
        assert(len(mainlist) == len(meterlist), "Number of main and meter channels should be equal")
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        # Get generators of timeseries
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)
        
        # Get a chunk of data
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])
        
        # Normalize 
        mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
        meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]
                
        X_batch= None
        Y_batch = None
        for i in range(num_meters):
            print(i)
            
            #Replace NaNs with 0s
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            #intersect chunks in order to have same length
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            #drop duplicates
            m1 = m1[~m1.index.duplicated()]
            mainchunks[i] = np.array(m1[ix])
            meterchunks[i] = np.array(m2[ix])
            #reshape and store to self attributes
            mainchunks[i] = np.reshape(mainchunks[i], (mainchunks[i].shape[0],1,1))            
            del m1, m2
            
            #concatenation
            if X_batch is None:
                X_batch = mainchunks[i]
                Y_batch = meterchunks[i]
            else:
                X_batch = np.append(X_batch, mainchunks[i], 0)
                Y_batch = np.append(Y_batch, meterchunks[i], 0)
            
        #for memory purposes    
        del mainchunks, meterchunks
         
        X_batch = np.reshape(X_batch, (X_batch.shape[0], 1, 1))
        self.mainchunk = X_batch
        self.meterchunk = Y_batch
        self.meterchunk = np.reshape(self.meterchunk, (len(self.meterchunk),-1 ))
        del X_batch, Y_batch
        
    @staticmethod
    def _normalize(chunk, mmax):
        '''Normalizes timeseries
        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries
        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk    
    
    def __len__(self):
        return len(self.mainchunk)
    
    def __getitem__(self, i):
        x = torch.from_numpy(self.mainchunk)
        y = torch.from_numpy(self.meterchunk)        
        return x[i], y[i]

    def __mmax__(self):
        return self.mmax
    
class TestDataset(Dataset):
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = np.reshape(Y, (len(Y),-1 ))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = torch.from_numpy(self.X)
        y = torch.from_numpy(self.Y)
        return x[i], y[i]
    
