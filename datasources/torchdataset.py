import torch
from abc import ABC
from typing import Iterator
from collections import deque
from torch.utils.data.dataset import T_co
from datasources.datasource import Datasource
from torch.utils.data import Dataset, IterableDataset
from datasources.preprocessing_lib import *
from lab.training_tools import ON_THRESHOLDS


class BaseElectricityDataset(ABC):

    def __init__(self, datasource, building, device, start_date,
                 end_date, rolling_window=True, window_size=50, mmax=None,
                 means=None, stds=None, meter_means=None, meter_stds=None,
                 sample_period=None, chunksize: int = 10000, shuffle=False):
        print('BaseElectricityDataset INIT')
        self.building = building
        self.device = device
        self.mmax = mmax
        self.means = means
        self.stds = stds
        self.meter_means = meter_means
        self.meter_stds = meter_stds
        self.chunksize = chunksize
        self.start_date = start_date
        self.end_date = end_date
        self.sample_period = sample_period
        self.datasource = datasource
        self.rolling_window = rolling_window
        self.window_size = window_size
        self.shuffle = shuffle
        self.threshold = ON_THRESHOLDS.get(device, 50)
        self.normalization_method = 'standardization'
        self.mainchunk = torch.tensor([])
        self.meterchunk = torch.tensor([])
        self.has_more_data = True
        self.__run__()

    def __run__(self):
        self._init_generators(datasource=self.datasource,
                              building=self.building,
                              device=self.device,
                              start_date=self.start_date,
                              end_date=self.end_date,
                              sample_period=self.sample_period,
                              chunksize=self.chunksize)
        self._reload()

    def __len__(self):
        return len(self.mainchunk)

    def _set_mmax(self, mainchunk):
        if self.mmax is None and len(mainchunk):
            self.mmax = mainchunk.max()

    def _set_means_stds(self, mainchunk, meterchunk):
        if self.means is None and self.stds is None and len(mainchunk):
            self.means = mainchunk.mean()
            self.stds = mainchunk.std()

        if self.meter_means is None and self.meter_stds is None and len(meterchunk):
            self.meter_means = meterchunk.mean()
            self.meter_stds = meterchunk.std()

    def __getitem__(self, i):
        x = self.mainchunk
        y = self.meterchunk
        return x[i].float(), y[i].float()

    def __mmax__(self):
        return self.mmax

    def _init_generators(self, datasource: Datasource, building, device, start_date,
                         end_date, sample_period, chunksize):
        self.datasource = datasource
        self.mains_generator = self.datasource.get_mains_generator(start=start_date,
                                                                   end=end_date,
                                                                   sample_period=sample_period,
                                                                   building=building,
                                                                   chunksize=chunksize)

        self.appliance_generator = self.datasource.get_appliance_generator(appliance=device,
                                                                           start=start_date,
                                                                           end=end_date,
                                                                           sample_period=sample_period,
                                                                           building=building,
                                                                           chunksize=chunksize)

    def _reload(self):
        try:
            mainchunk = next(self.mains_generator)
            meterchunk = next(self.appliance_generator)
            mainchunk, meterchunk = align_chunks(mainchunk, meterchunk)
            if len(mainchunk) or len(meterchunk):
                mainchunk, meterchunk = self._chunk_preprocessing(mainchunk, meterchunk)
                self.mainchunk, self.meterchunk = torch.from_numpy(np.array(mainchunk)), torch.from_numpy(
                    np.array(meterchunk))
            else:
                raise Exception('you need to increase chunksize')
        except StopIteration:
            self.has_more_data = False
            return

    def _chunk_preprocessing(self, mainchunk, meterchunk):
        mainchunk, meterchunk = replace_nans(mainchunk, meterchunk)
        if self.normalization_method == 'standardization':
            if None in [self.means, self.meter_means, self.meter_stds, self.stds]:
                self._set_means_stds(mainchunk, meterchunk)
            mainchunk, meterchunk = self._standardize_chunks(mainchunk, meterchunk)
        else:
            self._set_mmax(mainchunk)
            mainchunk, meterchunk = normalize_chunks(mainchunk, meterchunk, self.mmax)
        if self.rolling_window:
            mainchunk, meterchunk = apply_rolling_window(mainchunk, meterchunk, self.window_size)
        else:
            mainchunk, meterchunk = create_batches(mainchunk, meterchunk, self.window_size)
        if self.shuffle:
            mainchunk, meterchunk = mainchunk.sample(frac=1), meterchunk.sample(frac=1)
        return mainchunk, meterchunk

    def _standardize_chunks(self, mainchunk, meterchunk):
        ######
        # TODO: If chunk is a bad chunk (all zeros) then means/stds will be problematic
        ######
        if is_bad_chunk(mainchunk) or is_bad_chunk(meterchunk):
            print('chunks are all zeros')
            return mainchunk, meterchunk
        else:
            return standardize_chunks(mainchunk, meterchunk, self.means, self.stds, self.meter_means, self.meter_stds)


class ElectricityDataset(BaseElectricityDataset, Dataset):
    """ElectricityDataset dataset."""

    def __init__(self, datasource: Datasource, building, device, dates=None, rolling_window=True,
                 window_size=50, chunksize=10 ** 10, mmax=None, means=None, stds=None, meter_means=None,
                 meter_stds=None, sample_period=None):
        """
        Args:
            datasource(Datasource): datasource object, indicates to target dataset
            building(int): the desired building
            device(string): the desired device
            dates(list): list with the start and end(optional) dates for training window [start, end]
                        eg:['2016-04-01','2017-04-01']
        """
        print('ElectricityDataset INIT')
        super().__init__(datasource, building, device,
                         dates[0], dates[1], rolling_window, window_size,
                         mmax, means, stds, meter_means, meter_stds,
                         sample_period, chunksize)


class ElectricityMultiBuildingsDataset(BaseElectricityDataset, Dataset):
    """ElectricityMultiBuildingsDataset dataset."""

    def __init__(self, train_info=None, rolling_window=True, window_size=50, chunksize=10 ** 10, mmax=None, means=None,
                 stds=None, meter_means=None, meter_stds=None, sample_period=None, **load_kwargs):
        """
        Args:
            train_info(list): python list, contains to target datasets, dates,
                devices.
            ex: train_info = [{'device' : device,
                                 'datasource' : datasource,
                                 'building' : train_house,
                                 'train_dates' : train_dates,},
                              {'device' : device,
                                 'datasource' : datasource,
                                 'building' : train_house,
                                 'train_dates' : train_dates,},
                             ]
            building(int): the desired building
            device(string): the desired device
            dates(list): list with the start and end(optional) dates for training window [start, end]
                        eg:['2016-04-01','2017-04-01']
        """
        print('ElectricityMultiBuildingsDataset INIT')
        self.train_info = train_info
        super().__init__(datasource=None, building=None, device=None, start_date=None, end_date=None,
                         rolling_window=rolling_window, window_size=window_size, mmax=mmax, means=means,
                         stds=stds, meter_means=meter_means, meter_stds=meter_stds, sample_period=sample_period,
                         chunksize=chunksize, **load_kwargs)

    def __run__(self):
        num_buildings = len(self.train_info)
        self.mains_generators = [None] * num_buildings
        self.appliance_generators = [None] * num_buildings
        self.datasources = [None] * num_buildings
        self._init_generators(self.train_info, self.sample_period, self.chunksize)

    def _init_generators(self, train_info, sample_period, chunksize, **kwargs):
        for (index, element) in enumerate(train_info):
            datasource = element['datasource']
            building = element['building']
            device = element['device']
            start_date = element['dates'][0]
            end_date = element['dates'][1]
            self._init_single_building_generators(datasource, building, device, start_date,
                                                  end_date, sample_period, chunksize, index)

    def _init_single_building_generators(self, datasource: Datasource, building, device, start_date,
                                         end_date, sample_period, chunksize, index):
        self.datasources[index] = datasource
        self.mains_generators[index] = self.datasources[index].get_mains_generator(start=start_date,
                                                                                   end=end_date,
                                                                                   sample_period=sample_period,
                                                                                   building=building,
                                                                                   chunksize=chunksize)

        self.appliance_generators[index] = self.datasources[index].get_appliance_generator(appliance=device,
                                                                                           start=start_date,
                                                                                           end=end_date,
                                                                                           sample_period=sample_period,
                                                                                           building=building,
                                                                                           chunksize=chunksize)
        self.mains_generator, self.appliance_generator = self.mains_generators[index], self.appliance_generators[index]
        self._reload()

    def _reload(self):
        try:
            mainchunk = next(self.mains_generator)
            meterchunk = next(self.appliance_generator)
            mainchunk, meterchunk = align_chunks(mainchunk, meterchunk)
            if len(mainchunk) or len(meterchunk):
                mainchunk, meterchunk = self._chunk_preprocessing(mainchunk, meterchunk)
                mainchunk = torch.from_numpy(np.array(mainchunk))
                meterchunk = torch.from_numpy(np.array(meterchunk))
                self.mainchunk = torch.cat((self.mainchunk, mainchunk), 0)
                self.meterchunk = torch.cat((self.meterchunk, meterchunk), 0)
            else:
                raise Exception(' you need to increase chunksize')
        except StopIteration:
            return


class ElectricityIterableDataset(BaseElectricityDataset, IterableDataset):
    """ElectricityIterableDataset dataset."""

    def __init__(self, datasource: Datasource, building, device, dates=None, rolling_window=True,
                 window_size=50, mmax=None, means=None, stds=None, meter_means=None, meter_stds=None,
                 sample_period=None, chunksize: int = 10 ** 6, batch_size=32, shuffle=False):
        """
        Args:
            datasource(string): datasource object, indicates to target dataset
            building(int): the desired building
            device(string): the desired device
            dates(list): list with the start and end(optional) dates for training window [start, end]
                        eg:['2016-04-01','2017-04-01']

        Notes:
            a. It is recommended that the chuncksize is no less than 10**6
            b. For the pytorch dataloader to work properly with iterable datasets, shuffle must be False
        """
        print('ElectricityIterableDataset INIT')
        self.batch_size = batch_size
        self.data_len = None
        super().__init__(datasource, building, device,
                         dates[0], dates[1], rolling_window,
                         window_size, mmax, means, stds,
                         meter_means, meter_stds, sample_period, chunksize, shuffle)

    def __run__(self):
        self._calc_data_len()
        self._init_generators(datasource=self.datasource,
                              building=self.building,
                              device=self.device,
                              start_date=self.start_date,
                              end_date=self.end_date,
                              sample_period=self.sample_period,
                              chunksize=self.chunksize)
        self._reload()

    def __getitem__(self, index) -> T_co:
        pass

    def __len__(self):
        return self.data_len

    def _calc_data_len(self):
        print('#'*80)
        print('Calculating data length...')
        data_len = 0
        self._init_generators(datasource=self.datasource,
                              building=self.building,
                              device=self.device,
                              start_date=self.start_date,
                              end_date=self.end_date,
                              sample_period=self.sample_period,
                              chunksize=self.chunksize)
        has_data = True
        while has_data:
            try:
                mainchunk = next(self.mains_generator)
                meterchunk = next(self.appliance_generator)
                mainchunk, meterchunk = align_chunks(mainchunk, meterchunk)
                data_len += len(mainchunk)
                print(data_len)
            except StopIteration:
                print('Calculation Done!')
                has_data = False

        self.data_len = data_len
        print('data length: ', self.data_len)
        print('#' * 80)

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        return self._series_iterator(worker_info)

    def _series_iterator(self, worker_info):
        batch_size = self.batch_size
        if self._should_partition(worker_info):
            self._partition_chunks(worker_info)

        mainqueue = deque(self.mainchunk)
        meterqueue = deque(self.meterchunk)
        while True:
            mainval = mainqueue.popleft()
            meterval = meterqueue.popleft()
            if len(mainqueue) < batch_size and self.has_more_data:
                self._reload()
                if not self.mainchunk.nelement() or not self.meterchunk.nelement():
                    return
                if self._should_partition(worker_info):
                    self._partition_chunks(worker_info)
                mainqueue.extend(self.mainchunk)
                meterqueue.extend(self.meterchunk)
            yield mainval, meterval

    def _partition_chunks(self, worker_info):
        iter_start, iter_end = self._partition(worker_info, len(self.mainchunk))
        self.mainchunk = self.mainchunk[iter_start:iter_end]
        self.meterchunk = self.meterchunk[iter_start:iter_end]

    @staticmethod
    def _partition(worker_info, chunksize):
        partition_size = chunksize // worker_info.num_workers
        iter_start = worker_info.id * partition_size
        iter_end = min(iter_start + partition_size, chunksize)
        return iter_start, iter_end

    @staticmethod
    def _should_partition(worker_info):
        return worker_info is not None and worker_info.num_workers > 1


#
# class ElectricityIterableMultiBuildingsDataset(ElectricityMultiBuildingsDataset, IterableDataset):
#     """ElectricityIterableMultiBuildingsDataset dataset."""
#
#     def __init__(self, train_info=None, rolling_window=True, window_size=50, chunksize=10 ** 6, mmax=None, means=None,
#                  stds=None, meter_means=None, meter_stds=None, sample_period=None, batch_size=32, shuffle=False):
#         """
#         Args:
#             train_info(list): python list, contains to target datasets, dates,
#                 devices.
#             example=> train_info = [{'device' : device,
#                                      'datasource' : datasource,
#                                      'building' : train_house,
#                                      'train_dates' : train_dates,},
#                                    {'device' : device,
#                                      'datasource' : datasource,
#                                      'building' : train_house,
#                                      'train_dates' : train_dates,},
#                                   ]
#         """
#         self.batch_size = batch_size
#         super().__init__(train_info=train_info, rolling_window=rolling_window,
#                          window_size=window_size, mmax=mmax, means=means, stds=stds,
#                          meter_means=meter_means, meter_stds=meter_stds,
#                          sample_period=sample_period, chunksize=chunksize,
#                          shuffle=shuffle)
#
#     def __getitem__(self, index) -> T_co:
#         pass
#
#     def __len__(self):
#         return self.chunksize
#
#     def __iter__(self) -> Iterator[T_co]:
#         worker_info = torch.utils.data.get_worker_info()
#         self._reload()
#         return self._series_iterator(worker_info)
#
#     def _init_single_building_generators(self, datasource: Datasource, building, device, start_date,
#                                          end_date, sample_period, chunksize, index):
#         chunksize = chunksize // len(self.train_info)
#         self.datasources[index] = datasource
#         self.mains_generators[index] = self.datasources[index].get_mains_generator(start=start_date,
#                                                                                    end=end_date,
#                                                                                    sample_period=sample_period,
#                                                                                    building=building,
#                                                                                    chunksize=chunksize)
#
#         self.appliance_generators[index] = self.datasources[index].get_appliance_generator(appliance=device,
#                                                                                            start=start_date,
#                                                                                            end=end_date,
#                                                                                            sample_period=sample_period,
#                                                                                            building=building,
#                                                                                            chunksize=chunksize)
#
#     def _reload(self):
#         for (index, element) in enumerate(self.train_info):
#             try:
#                 print('#'*80)
#                 print('ELEMENT: ', index, element)
#                 mainchunk = next(self.mains_generators[index])
#                 meterchunk = next(self.appliance_generators[index])
#                 mainchunk, meterchunk = align_chunks(mainchunk, meterchunk)
#                 print(mainchunk.shape)
#                 print('#' * 80)
#                 if len(mainchunk) or len(meterchunk):
#                     mainchunk, meterchunk = self._chunk_preprocessing(mainchunk, meterchunk)
#                     mainchunk = torch.from_numpy(np.array(mainchunk))
#                     meterchunk = torch.from_numpy(np.array(meterchunk))
#                     self.mainchunk = torch.cat((self.mainchunk, mainchunk), 0)
#                     self.meterchunk = torch.cat((self.meterchunk, meterchunk), 0)
#                 else:
#                     raise Exception('you need to increase chunksize')
#             except StopIteration:
#                 print('no more data to load')
#         return
#
#     def _series_iterator(self, worker_info):
#         batch_size = self.batch_size
#         if self._should_partition(worker_info):
#             self._partition_chunks(worker_info)
#
#         mainqueue = deque(self.mainchunk)
#         meterqueue = deque(self.meterchunk)
#         while True:
#             mainval = mainqueue.popleft()
#             meterval = meterqueue.popleft()
#             if len(mainqueue) < batch_size:
#                 self._reload()
#                 if not self.mainchunk.nelement() or not self.meterchunk.nelement():
#                     return
#                 if self._should_partition(worker_info):
#                     self._partition_chunks(worker_info)
#                 mainqueue.extend(self.mainchunk)
#                 meterqueue.extend(self.meterchunk)
#             yield mainval, meterval
#
#     def _partition_chunks(self, worker_info):
#         iter_start, iter_end = self._partition(worker_info, len(self.mainchunk))
#         self.mainchunk = self.mainchunk[iter_start:iter_end]
#         self.meterchunk = self.meterchunk[iter_start:iter_end]
#
#     @staticmethod
#     def _partition(worker_info, chunksize):
#         partition_size = chunksize // worker_info.num_workers
#         iter_start = worker_info.id * partition_size
#         iter_end = min(iter_start + partition_size, chunksize)
#         return iter_start, iter_end
#
#     @staticmethod
#     def _should_partition(worker_info):
#         return worker_info is not None and worker_info.num_workers > 1

