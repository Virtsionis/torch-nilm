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
    """
    -BaseElectricityDataset

    This is a base class containing all the methods necessary for loading, alignment and preprocessing of NILM data.
    In the current version, NILMTK supported datasets were used https://arxiv.org/abs/1404.3878.
    Thus, the part of loading the data depends on NILMTK package. In the future, the class will support dataset from
    other sources (e.g. csv, h5 etc).

    Args:
        datasource(datasource): datasource object, indicates the target datasource to load the data from
        building(int): the desired building
        device(string): the target electrical appliance
        dates(list): list with the start and end(optional) dates for training window [start, end]
                    eg:['2016-04-01','2017-04-01']
        rolling_window(bool): corresponds to the preprocessing method which is the rolling_window described in:
            https://dl.acm.org/doi/abs/10.1145/3200947.3201011
            If given False, preprocessing is sequence-to-sequence type. Default value is True
        window_size(int): the size of the rolling window, has effect if rolling_window=True
        chunksize(int): the size of loaded chunk from NILMTK generators
        mmax(float): the maximum value of mains time series,
            needed for the de-normalization of the data
        means(float): the mean value of mains time series
            needed for the de-standardization of the data
        stds(float): the std value of mains time series
            needed for the de-standardization of the data
        meter_means(float): the mean value of meter time series
            needed for the de-standardization of the data
        meter_stds(float): the std value of meter time series
            needed for the de-standardization of the data
        sample_period(int): the sample period given in seconds
            if sample_period is larger than the sampling of the data, then NILMTK downsamples the measurements.
            Else, upsampling is excecuted
        normalization_method(str): the normalization method of the time series
            possible values: 'standardization' or 'normalization'
            if 'standardization' is given, the time series are standardized with mean & std values
                of the mains & target meter time series
            if 'normalization' is given, the time series are normalized with the max value of the mains time series

    Functionality in a nut-shell:
        After saving the input arguments as class properties, the NILMTK generators are initialized and
        all the data are loaded in the memory. Then, the mains & target meter time series are aligned before
        preprocessing takes place. The preprocessing consists of time series normalization/standardization
        (depends on the chosen normalization method) and the creation of windows if rolling window is True.
        The feeding of the preprocessed data is taken care of the pytorch dataloader.
    """
    def __init__(self, datasource, building, device, start_date,
                 end_date, rolling_window=True, window_size=50, mmax=None,
                 means=None, stds=None, meter_means=None, meter_stds=None,
                 sample_period=None, chunksize: int = 10000, shuffle=False,
                 normalization_method='standardization'):
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
        self.normalization_method = normalization_method
        self.mainchunk = torch.tensor([])
        self.meterchunk = torch.tensor([])
        self.has_more_data = True
        self._run()

    def _run(self):
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
        elif self.normalization_method == 'normalization':
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
            return mainchunk, meterchunk
        else:
            return standardize_chunks(mainchunk, meterchunk, self.means, self.stds, self.meter_means, self.meter_stds)


class ElectricityDataset(BaseElectricityDataset, Dataset):
    """
    -ElectricityDataset

    The purpose of this class is to load, align and preprocess the requested data in a proper format for
    pytorch dataloaders to handle. In the current version, NILMTK supported datasets were used
    https://arxiv.org/abs/1404.3878. Thus, the part of loading the data depends on NILMTK package.
    In the future, the class will support dataset from other sources (e.g. csv, h5 etc).
    It should be noted that all the ElectricityDataset was written according the pytorch documentation guidelines:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        datasource(datasource): datasource object, indicates the target datasource to load the data from
        building(int): the desired building
        device(string): the target electrical appliance
        dates(list): list with the start and end(optional) dates for training window [start, end]
                    eg:['2016-04-01','2017-04-01']
        rolling_window(bool): corresponds to the preprocessing method which is the rolling_window described in:
            https://dl.acm.org/doi/abs/10.1145/3200947.3201011
            If given False, preprocessing is sequence-to-sequence type. Default value is True
        window_size(int): the size of the rolling window, has effect if rolling_window=True
        chunksize(int): the size of loaded chunk from NILMTK generators
        mmax(float): the maximum value of mains time series,
            needed for the de-normalization of the data
        means(float): the mean value of mains time series
            needed for the de-standardization of the data
        stds(float): the std value of mains time series
            needed for the de-standardization of the data
        meter_means(float): the mean value of meter time series
            needed for the de-standardization of the data
        meter_stds(float): the std value of meter time series
            needed for the de-standardization of the data
        sample_period(int): the sample period given in seconds
            if sample_period is larger than the sampling of the data, then NILMTK downsamples the measurements.
            Else, upsampling is excecuted
        normalization_method(str): the normalization method of the time series
            possible values: 'standardization' or 'normalization'
            if 'standardization' is given, the time series are standardized with mean & std values
                of the mains & target meter time series
            if 'normalization' is given, the time series are normalized with the max value of the mains time series

    Functionality in a nut-shell:
        After saving the input arguments as class properties, the NILMTK generators are initialized and
        all the data are loaded in the memory. Then, the mains & target meter time series are aligned before
        preprocessing takes place. The preprocessing consists of time series normalization/standardization
        (depends on the chosen normalization method) and the creation of windows if rolling window is True.
        The feeding of the preprocessed data is taken care of the pytorch dataloader.

    Example of use:
        train_dataset_all = ElectricityDataset(datasource=datasource, building=int(train_house), window_size=WINDOW,
                                               device=device, dates=train_dates, sample_period=SAMPLE_PERIOD)

        train_size = int(0.8 * len(train_dataset_all))
        val_size = len(train_dataset_all) - train_size
        train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=BATCH,
                                  shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH,
                                shuffle=False, num_workers=8)

        trainer.fit(model, train_loader, val_loader)
    """
    def __init__(self, datasource: Datasource, building, device, dates=None, rolling_window=True,
                 window_size=50, chunksize=10 ** 10, mmax=None, means=None, stds=None, meter_means=None,
                 meter_stds=None, sample_period=None, normalization_method='standardization'):
        super().__init__(datasource, building, device,
                         dates[0], dates[1], rolling_window, window_size,
                         mmax, means, stds, meter_means, meter_stds,
                         sample_period, chunksize, normalization_method=normalization_method, )


class ElectricityMultiBuildingsDataset(BaseElectricityDataset, Dataset):
    """
    ElectricityMultiBuildingsDataset

    The purpose of this class is to load, align and preprocess the requested data in a proper format for
    pytorch dataloaders to handle. Specifically, this class manipulates data measurements from a set of different
    buildings, in order training on data from multiple datasources is possible as described in:
        https://link.springer.com/chapter/10.1007/978-3-030-20257-6_2.
    MultiBuilding training is helpful in order to test the generalization capabilities of the models.

    In the current version, NILMTK supported datasets were used.
    Thus, the part of loading the data depends on NILMTK package. In the future, the class will support
    dataset from other sources (e.g. csv, h5 etc)

    Args:
        train_info(list): python list, contains target datasets, dates, devices.
        ex: train_info = [{'device' : device,
                             'datasource' : datasource,
                             'building' : train_house,
                             'train_dates' : train_dates,},
                          {'device' : device,
                             'datasource' : datasource,
                             'building' : train_house,
                             'train_dates' : train_dates,},
                         ]
        datasource(datasource): datasource object, indicates the target datasource to load the data from
        building(int): the desired building
        device(string): the desired electrical appliance
        train_dates(list): list with the start and end(optional) dates for training window [start, end]
                    eg:['2016-04-01','2017-04-01']
        rolling_window(bool): corresponds to the preprocessing method which is the rolling_window described in:
            https://dl.acm.org/doi/abs/10.1145/3200947.3201011
            If given False, preprocessing is sequence-to-sequence type. Default value is True
        window_size(int): the size of the rolling window, has effect if rolling_window=True
        chunksize(int): the size of loaded chunk from NILMTK generators
        mmax(float): the maximum value of mains time series,
            needed for the de-normalization of the data
        means(float): the mean value of mains time series
            needed for the de-standardization of the data
        stds(float): the std value of mains time series
            needed for the de-standardization of the data
        meter_means(float): the mean value of meter time series
            needed for the de-standardization of the data
        meter_stds(float): the std value of meter time series
            needed for the de-standardization of the data
        sample_period(int): the sample period given in seconds
            if sample_period is larger than the sampling of the data, then NILMTK downsamples the measurements.
            Else, upsampling is excecuted

    Functionality in a nut-shell:
        After saving the input arguments as class properties, the NILMTK generators are initialized and
        all the data are loaded in the memory. Due to the fact that the data should be loaded from multiple
        datasources, two lists are instantiated containing the corresponding data generators. Thus, the reloading
        of the data is different in comparison to the ElectricityDataset. The class attribute self.mains_generators
        is a list that contains all the generators needed for the multi-building training. In method _run, this list
        is initialized to have length equal to the number of buildings. Hence, the length of the list is the same as
        the number of buildings-generators specified in the train_info of the experiment. The method _init_generators
        fills each position of the self.mains_generators list with the specific building generator by calling the method
         _init_single_building_generators. Essentially, the _init_generators is a loop.

        Then, the mains & target meter time series are aligned before preprocessing takes place. The preprocessing
        consists of time series normalization/standardization (depends on the chosen normalization method) and the
        creation of windows if rolling window is True. The feeding of the preprocessed data is taken care of the pytorch
        dataloader.

    Example of use:
        train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
                                                             window_size=WINDOW,
                                                             sample_period=SAMPLE_PERIOD)
        train_size = int(0.8 * len(train_dataset_all))
        val_size = len(train_dataset_all) - train_size
        train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=BATCH,
                                  shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH,
                                shuffle=False, num_workers=8)

        trainer.fit(model, train_loader, val_loader)

    """
    def __init__(self, train_info=None, rolling_window=True, window_size=50, chunksize=10 ** 10, mmax=None, means=None,
                 stds=None, meter_means=None, meter_stds=None, sample_period=None,
                 normalization_method='standardization', **load_kwargs):
        self.train_info = train_info
        super().__init__(datasource=None, building=None, device=None, start_date=None, end_date=None,
                         rolling_window=rolling_window, window_size=window_size, mmax=mmax, means=means,
                         stds=stds, meter_means=meter_means, meter_stds=meter_stds, sample_period=sample_period,
                         chunksize=chunksize, normalization_method=normalization_method, **load_kwargs)

    def _run(self):
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
    """
    ElectricityIterableDataset

    The purpose of this class is to load, align and preprocess the requested data in a proper format for
    pytorch dataloaders to handle, in a more efficient way than the ElectricityDataset. Specifically, the
    main problem with ElectricityDataset class is that the data has to be loaded in memory completely and
    then the pytorch dataloaders feed it to the models, an inefficient case for low-RAM machines.
    To fight this problem, an iterable-style dataset was developed following the pytorch documentation guidelines:
    https://pytorch.org/docs/stable/data.html#iterable-style-datasets.
    In order for iterable datasets to work smoothly with pytorch dataloaders, the length of the dataset must be
    known a priory. Due to the lack of guidelines in the pytorch documentation about this matter, we decided to
    create the method _calc_data_len() to specify the number of points of the dataset. This method loads and aligns
    the data in the same manner as the method _reload, without the preprocessing. Hence, the actual length of the
    iterable dataset is calculated. After this process is finished, the data generators have to be re-initialized.
    Thus, _init_generators is re-called. Yet, this not an optimal solution, but it works for the time being.

    In the current version, NILMTK supported datasets were used
    https://arxiv.org/abs/1404.3878. Thus, the part of loading the data depends on NILMTK package.
    In the future, the class will support dataset from other sources (e.g. csv, h5 etc).
    It should be noted that all the ElectricityDataset was written according the pytorch documentation guidelines:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        datasource(datasource): datasource object, indicates the target datasource to load the data from
        building(int): the desired building
        device(string): the desired device
        dates(list): list with the start and end(optional) dates for training window [start, end]
                    eg:['2016-04-01','2017-04-01']
        rolling_window(bool): corresponds to the preprocessing method which is the rolling_window described in:
            https://dl.acm.org/doi/abs/10.1145/3200947.3201011
            If given False, preprocessing is sequence-to-sequence type. Default value is True
        window_size(int): the size of the rolling window, has effect if rolling_window=True
        chunksize(int): the size of loaded chunk from NILMTK generators
        mmax(float): the maximum value of mains time series,
            needed for the de-normalization of the data
        means(float): the mean value of mains time series
            needed for the de-standardization of the data
        stds(float): the std value of mains time series
            needed for the de-standardization of the data
        meter_means(float): the mean value of meter time series
            needed for the de-standardization of the data
        meter_stds(float): the std value of meter time series
            needed for the de-standardization of the data
        sample_period(int): the sample period given in seconds
            if sample_period is larger than the sampling of the data, then NILMTK downsamples the measurements.
            Else, upsampling is excecuted
        normalization_method(str): the normalization method of the time series
            possible values: 'standardization' or 'normalization'
            if 'standardization' is given, the time series are standardized with mean & std values
                of the mains & target meter time series
            if 'normalization' is given, the time series are normalized with the max value of the mains time series
        batch_size(int): the batch size needed for the series iterator to work
            Default: 32

    Functionality in a nut-shell:
        After saving the input arguments as class properties, the NILMTK generators are initialized for the
        first time and the length of the dataset is calculated. Then, the generators are re-initialized and
        the first chunk of data is loaded in the memory. The reload and preprocessing methods are the same
        as the ElectricityDataset. Through the method 'series_iterator', the data keeps reloading until the
        generators are exhausted.

    Example of use:
        train_dataset = ElectricityIterableDataset(datasource=datasource,
                                                   building=int(train_house),
                                                   window_size=WINDOW,
                                                   device=device,
                                                   dates=train_dates,
                                                   sample_period=SAMPLE_PERIOD,
                                                   rolling_window=True,
                                                  )
        train_loader = DataLoader(train_dataset, batch_size=BATCH,
                                  shuffle=True, num_workers=8)
        trainer.fit(model, train_loader)

    Important Notes:
        a. It is recommended that the chuncksize is no less than 10**6
        b. For the pytorch dataloader to work properly with iterable datasets, shuffle must be False
        c. Iterable datasets don't support dataloader with shuffle=True
    """
    def __init__(self, datasource: Datasource, building, device, dates=None, rolling_window=True,
                 window_size=50, mmax=None, means=None, stds=None, meter_means=None, meter_stds=None,
                 sample_period=None, chunksize: int = 10 ** 6, batch_size=32, shuffle=False,
                 normalization_method='standardization'):
        self.batch_size = batch_size
        self.data_len = None
        super().__init__(datasource, building, device,
                         dates[0], dates[1], rolling_window,
                         window_size, mmax, means, stds,
                         meter_means, meter_stds, sample_period,
                         chunksize, shuffle, normalization_method=normalization_method)

    def _run(self):
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
            except StopIteration:
                has_data = False

        self.data_len = data_len

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

