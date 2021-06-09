import torch
import numpy as np
from torch.utils.data import Dataset
from nilmtk import DataSet as NILMTKDataSet

class MyChunk(Dataset):
    """MyChunk dataset."""

    def __init__(self, path, building, device,  dates=None, transform=None,
                 window_size=50, test=False, mmax=None, sample_period=None,**load_kwargs):
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
        self.test = test

        #loads h5 file with Nilmtk DataSet class
        self.dataset = NILMTKDataSet(path)

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
        self.elec = elec1
        #get meter metadata
        self.meter_metadata = meter

        #these are nilmtk generators
        if sample_period:
            self.main_power_series = mains.power_series(**load_kwargs, sample_period=sample_period)
            self.meter_power_series = meter.power_series(**load_kwargs, sample_period=sample_period)
        else:
            self.main_power_series = mains.power_series(**load_kwargs)
            self.meter_power_series = meter.power_series(**load_kwargs)

        #get chunks
        mainchunk = next(self.main_power_series)
        meterchunk = next(self.meter_power_series)

        #get chunk details, but not needed
        self.timeframe = mainchunk.timeframe
        self.chunk_name = mainchunk.name

        mainchunk = mainchunk[~mainchunk.index.duplicated()]
        meterchunk = meterchunk[~meterchunk.index.duplicated()]

        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        #normalize chunks
        if self.mmax == None:
            self.mmax = mainchunk.max()
        mainchunk = mainchunk / self.mmax
        meterchunk = meterchunk / self.mmax

        #Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)

        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk)-self.window_size+1)[:, None]
        # time_indexes are not needed
        mains_time_index = mainchunk.index
        meter_time_index = meterchunk.index

        mainchunk, mains_time_index = mainchunk[indexer], mains_time_index[indexer]
        meterchunk, meter_time_index = meterchunk[self.window_size-1:], meter_time_index[self.window_size-1:]

        # mainchunk = np.reshape(mainchunk, (mainchunk.shape[0], 1, mainchunk.shape[1]))
        self.mainchunk = np.array(mainchunk)
        self.mains_time_index = np.array(mains_time_index)
        self.meterchunk = np.array(meterchunk)
        self.meter_time_index = np.array(meter_time_index)

    def _preprocessing(self):
        pass

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

    def __init__(self, device, transform=None,filename=None,
                 window_size=50, mmax=None, sample_period=None,**load_kwargs):
        """
        Args:
            filename (string): name of the train info file.
            device(string): the desired device
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.mmax = None
        self.device = device
        self.window_size = window_size
        self.transform = transform

        meterlist, mainlist = self.load_from_file(filename)

        meterps, mainps, num_meters = self.get_generators(meterlist, mainlist,
                                                         sample_period=sample_period,
                                                         **load_kwargs)

        # theoritically from here all should be called in _getitem_
        # Get a chunk of data
        # if chunksize not given it loads all data on memory
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])

        # preprocess current chunk
        self.mainchunk, self.meterchunk = self.preprocessing(meterchunks, mainchunks, num_meters)

    def __len__(self):
        return len(self.mainchunk)

    def __getitem__(self, i):
        x = torch.from_numpy(self.mainchunk)
        y = torch.from_numpy(self.meterchunk)
        return x[i].float(), y[i].float()

    def __mmax__(self):
        return self.mmax

    def preprocessing(self, meterchunks, mainchunks, num_meters):
        # Normalize
        mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
        meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

        X_batch = None
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

            #concatenation
            if X_batch is None:
                X_batch = mainchunks[i]
                Y_batch = meterchunks[i]
            else:
                X_batch = np.append(X_batch, mainchunks[i], 0)
                Y_batch = np.append(Y_batch, meterchunks[i], 0)

        X_batch = np.squeeze(X_batch)
        Y_batch = np.squeeze(Y_batch)

        return self.time_windows(X_batch, Y_batch)

    def time_windows(self, x, y):
        '''
        forms the dataset time windows
        returns np arrays
        '''
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(x) - self.window_size + 1)[:, None]
        x = x[indexer]
        y = y[self.window_size - 1:]
        return np.array(x), np.array(y)

    def load_from_file(self, filename):
        '''
        reads specific buildings and timeframes from the given file
        returns the corresponding meter-mains lists
        '''
        meterlist = []
        mainlist = []

        if filename == None:
            file = open('baseTrainSetsInfo_' + self.device, 'r')
        else:
            file = open(filename, 'r')

        for line in file:
            #take tokens from file
            toks = line.split(',')

            #loads h5 file with Nilmtk DataSet class
            train = NILMTKDataSet(toks[0])

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
        return meterlist, mainlist

    def get_generators(self, meterlist, mainlist, sample_period, **load_kwargs):
        assert(len(mainlist) == len(meterlist), "Number of main and meter channels should be equal")
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters

        # Get generators of timeseries
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs, sample_period=sample_period)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs, sample_period=sample_period)
        return meterps, mainps, num_meters

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
