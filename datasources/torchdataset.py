import itertools
import math
from typing import Iterator

import torch
import numpy as np
from collections import deque
from torch.utils.data import Dataset, IterableDataset
from nilmtk import DataSet as NILMTKDataSet
from torch.utils.data.dataset import T_co

from datasources.datasource import Datasource
from exceptions.lab_exceptions import NotSupportingMultiprocessing, InvalidIteratorArgs
from multiprocessing import Queue


class ElectricityDataset(IterableDataset):
    """MyChunk dataset."""

    def __init__(self, datasource: Datasource, building, device,
                 start_date: str, end_date: str, transform=None,
                 window_size=50, test=False, mmax=None, sample_period=None, chunksize: int = 10000,
                 batch_size=32,
                 **load_kwargs):
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
        self.building = building
        self.device = device
        self.transform = transform
        self.test = test
        self.mmax = mmax
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.start_date = start_date
        self.end_date = end_date
        self.sample_period = sample_period

        self.datasource = datasource
        self.window_size = window_size

        self._init_generators(building, chunksize, device, start_date, end_date, sample_period)

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        self._init_generators(self.building, self.chunksize, self.device, self.start_date, self.end_date,
                              self.sample_period)
        return self._series_iterator(worker_info)

    def __getitem__(self, index) -> T_co:
        pass

    def _series_iterator(self, worker_info):
        batch_size = self.batch_size
        if self._should_partition(worker_info):
            self._partition_chunks(worker_info)

        mainqueue = deque(self.mainchunk)
        meterqueue = deque(self.meterchunk)
        while True:
            mainval = mainqueue.popleft()
            meterval = meterqueue.popleft()
            if len(mainqueue) < batch_size:
                self._reload()
                if not self.mainchunk.nelement() or not self.meterchunk.nelement():
                    return
                if self._should_partition(worker_info):
                    self._partition_chunks(worker_info)
                mainqueue.extend(self.mainchunk)
                meterqueue.extend(self.meterchunk)
            yield mainval, meterval

    def _partition(self, worker_info, chunksize):
        partition_size = chunksize // worker_info.num_workers
        iter_start = worker_info.id * partition_size
        iter_end = min(iter_start + partition_size, chunksize)
        return iter_start, iter_end

    def _partition_chunks(self, worker_info):
        iter_start, iter_end = self._partition(worker_info, len(self.mainchunk))
        self.mainchunk = self.mainchunk[iter_start:iter_end]
        self.meterchunk = self.meterchunk[iter_start:iter_end]

    def _should_partition(self, worker_info):
        return worker_info is not None and worker_info.num_workers > 1

    def _reload(self):
        self.mainchunk = torch.tensor([])
        self.meterchunk = torch.tensor([])
        try:
            mainchunk = next(self.mains_generator)
            meterchunk = next(self.appliance_generator)

            mainchunk, meterchunk = self._align_chunks(mainchunk, meterchunk)
            mainchunk, meterchunk = self._normalize_chunks(mainchunk, meterchunk)
            mainchunk, meterchunk = self._replace_nans(mainchunk, meterchunk)
            mainchunk, meterchunk = self._apply_rolling_window(mainchunk, meterchunk)
            self.mainchunk, self.meterchunk = torch.from_numpy(np.array(mainchunk)), torch.from_numpy(
                np.array(meterchunk))
        except StopIteration:
            return

    def _init_generators(self, building, chunksize, device, start_date, end_date, sample_period):
        self.mains_generator = self.datasource.get_mains_generator(start=start_date, end=end_date,
                                                                   sample_period=sample_period,
                                                                   building=building, chunksize=chunksize)
        self.appliance_generator = self.datasource.get_appliance_generator(appliance=device,
                                                                           start=start_date,
                                                                           end=end_date,
                                                                           sample_period=sample_period,
                                                                           building=building,
                                                                           chunksize=chunksize)
        self._reload()

    def _apply_rolling_window(self, mainchunk, meterchunk):
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk) - self.window_size + 1)[:, None]
        mainchunk = mainchunk[indexer]
        meterchunk = meterchunk[self.window_size - 1:]
        return mainchunk, meterchunk

    def _replace_nans(self, mainchunk, meterchunk):
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        return mainchunk, meterchunk

    def _normalize_chunks(self, mainchunk, meterchunk):
        if self.mmax is None:
            self.mmax = mainchunk.max()
        mainchunk = mainchunk / self.mmax
        meterchunk = meterchunk / self.mmax
        return mainchunk, meterchunk

    def _align_chunks(self, mainchunk, meterchunk):
        mainchunk = mainchunk[~mainchunk.index.duplicated()]
        meterchunk = meterchunk[~meterchunk.index.duplicated()]
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]
        return mainchunk, meterchunk

    def __mmax__(self):
        return self.mmax
