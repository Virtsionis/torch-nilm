import os
from typing import Union

import torch
import pandas as pd

from callbacks.callbacks_factories import TrainerCallbacksFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset, ElectricityMultiBuildingsDataset, ElectricityIterableDataset
from modules.helpers import create_tree_dir, create_time_folds
from modules.nilm_trainer import train_eval
from torch.utils.data import DataLoader, random_split
from constants.constants import *
from constants.device_windows import WINDOWS
from constants.enumerates import SupportedNilmExperiments, ElectricalAppliances, SupportedExperimentCategories, \
    StatMeasures, SupportedExperimentVolumes
from modules.reporting import get_final_report, get_statistical_report

with torch.no_grad():
    torch.cuda.empty_cache()


class NILMExperiments:

    def __init__(self, project_name: str = None, clean_project: bool = False, experiment_categories: list = None,
                 devices: list = None, experiment_volume: str = SupportedExperimentVolumes.LARGE_VOLUME,
                 save_timeseries_results: bool = True, inference_cpu: bool = False, data_dir: str = None,
                 train_file_dir: str = None, test_file_dir: str = None, experiment_type: str = None,
                 train_params: dict = None, model_hparams: dict = None,
                 ):

        self.project_name = project_name
        self.clean_project = clean_project
        self.save_timeseries = save_timeseries_results
        self.inference_cpu = inference_cpu
        self.devices = devices
        self.model_hparams = model_hparams
        self.experiment_type = experiment_type
        self.train_params = train_params
        self._set_models()
        self._set_train_parameters(train_params)
        self._set_supported_experiments()
        self._set_experiment_categories(experiment_categories)
        self._set_data_dir(data_dir)
        self._set_experiment_volume(experiment_volume)
        self._set_train_test_file_dir(train_file_dir, test_file_dir)
        self._create_project_structure()

    def _set_models(self):
        if self.model_hparams and len(self.model_hparams):
            self.models = self.model_hparams.keys()
        else:
            raise Exception('No models or model hyper parameters are defined')

    def _set_supported_experiments(self):
        self.experiments = {
            SupportedNilmExperiments.BENCHMARK: self.run_benchmark,
            SupportedNilmExperiments.CROSS_VALIDATION: self.run_cross_validation,
            SupportedNilmExperiments.HYPERPARAM_TUNE_CV: self.run_hyperparameter_tuning_cross_validation,
        }

    def get_supported_experiments(self):
        """
        returns the supported nilm experiments
        """
        return [experiment.name for experiment in self.experiments.keys()]

    def _set_default_train_parameters(self):
        self.epochs = 100
        self.iterations = 5
        self.sample_period = 6
        self.batch_size = 256
        self.iterable_dataset = False
        self.fixed_window = 100
        self.train_test_split = 0.8
        self.cv_folds = 3

    def _set_train_parameters(self, train_params: dict = None):
        if train_params:
            self.epochs = self.train_params[EPOCHS]
            self.iterations = self.train_params[ITERATIONS]
            self.sample_period = self.train_params[SAMPLE_PERIOD]
            self.batch_size = self.train_params[BATCH_SIZE]
            self.iterable_dataset = self.train_params[ITERABLE_DATASET]
            self.fixed_window = self.train_params[FIXED_WINDOW]
            self.train_test_split = self.train_params[TRAIN_TEST_SPLIT]
            self.cv_folds = self.train_params[CV_FOLDS]
        else:
            self._set_default_train_parameters()

    def _set_experiment_categories(self, experiment_categories: list = None):
        if experiment_categories:
            self.experiment_categories = [experiment.value for experiment in experiment_categories
                                          if experiment.value in [SupportedExperimentCategories.SINGLE_CATEGORY.value,
                                                                  SupportedExperimentCategories.MULTI_CATEGORY.value]]
        else:
            self.experiment_categories = [SupportedExperimentCategories.SINGLE_CATEGORY,
                                          SupportedExperimentCategories.MULTI_CATEGORY]

    def _set_data_dir(self, data_dir: str = None,):
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = DIR_DATASETS

    def _set_experiment_volume(self, experiment_volume: str = None):
        if experiment_volume:
            self.experiment_volume = experiment_volume.value
        else:
            self.experiment_volume = SupportedExperimentVolumes.LARGE_VOLUME.value

    def _set_train_test_file_dir(self, train_file_dir: str = None, test_file_dir: str = None):
        if train_file_dir and os.path.isdir(train_file_dir):
            self.train_file_dir = train_file_dir
        else:
            self.train_file_dir = '/'.join([DIR_BENCHMARK_NAME, self.experiment_volume, DIR_TRAIN_NAME, ''])

        if test_file_dir and os.path.isdir(test_file_dir):
            self.test_file_dir = test_file_dir
        else:
            self.test_file_dir = '/'.join([DIR_BENCHMARK_NAME, self.experiment_volume, DIR_TEST_NAME, ''])

    def _create_project_structure(self):
        experiment_categories = self.experiment_categories
        devices = self.devices
        models = self.model_hparams.keys()
        clean_project = self.clean_project
        project_name = self.project_name
        tree_levels = {ROOT_LEVEL: project_name,
                       LEVEL_1_NAME: [DIR_RESULTS_NAME],
                       LEVEL_2_NAME: devices,
                       LEVEL_3_NAME: models,
                       EXPERIMENTS_NAME: experiment_categories}
        create_tree_dir(tree_levels=tree_levels, clean=clean_project)
        self.tree_levels = tree_levels

    def _prepare_cv_parameters(self, experiment_category: str = None, device: str = None, window: int = None):
        if not experiment_category:
            experiment_category = SupportedExperimentCategories.SINGLE_CATEGORY.value
        try:
            file = open('{}base{}TrainSetsInfo_{}'.format(self.train_file_dir, experiment_category, device), 'r')
        except Exception as e:
            raise e
        train_set, dates, train_house = None, None, None
        for line in file:
            toks = line.split(',')
            train_set = toks[0]
            train_house = int(toks[1])
            dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
            break
        file.close()
        if train_set and dates and train_house:
            datasource = DatasourceFactory.create_datasource(train_set)
            time_folds = create_time_folds(start_date=dates[0], end_date=dates[1],
                                           folds=self.cv_folds, drop_last=False)
            return datasource, time_folds, train_set, train_house
        else:
            raise Exception('Not a proper train file')

    def _prepare_cv_dataset(self, device, fold, window, datasource, time_folds, train_set, train_house):

        train_dates = time_folds[fold][TRAIN_DATES]
        train_info = []
        for train_date in train_dates:
            if len(train_date):
                train_info.append({
                    COLUMN_DEVICE: device,
                    COLUMN_DATASOURCE: datasource,
                    COLUMN_BUILDING: int(train_house),
                    COLUMN_DATES: train_date,
                })
        train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
                                                             window_size=window,
                                                             sample_period=self.sample_period)
        return train_dataset_all

    def _prepare_train_dataset(self, experiment_category: str = None, device: str = None, window: int = None):
        file = open('{}base{}TrainSetsInfo_{}'.format(self.train_file_dir, experiment_category, device), 'r')
        train_info = []
        for line in file:
            toks = line.split(',')
            train_set = toks[0]
            train_house = toks[1]
            train_dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
            datasource = DatasourceFactory.create_datasource(train_set)
            if experiment_category == SupportedExperimentCategories.MULTI_CATEGORY:
                train_info.append({
                    COLUMN_DEVICE: device,
                    COLUMN_DATASOURCE: datasource,
                    COLUMN_BUILDING: int(train_house),
                    COLUMN_DATES: train_dates,
                })
            else:
                file.close()
                if self.iterable_dataset:
                    train_dataset_all = ElectricityIterableDataset(datasource=datasource,
                                                                   building=int(train_house),
                                                                   window_size=window,
                                                                   device=device,
                                                                   dates=train_dates,
                                                                   sample_period=self.sample_period)
                else:
                    train_dataset_all = ElectricityDataset(datasource=datasource,
                                                           building=int(train_house),
                                                           window_size=window,
                                                           device=device,
                                                           dates=train_dates,
                                                           sample_period=self.sample_period)

                return train_dataset_all
        file.close()
        train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
                                                             window_size=window,
                                                             sample_period=self.sample_period)
        return train_dataset_all

    def _prepare_train_val_loaders(self, train_dataset_all: Union[ElectricityDataset,
                                                                  ElectricityMultiBuildingsDataset,
                                                                  ElectricityIterableDataset] = None):
        if train_dataset_all:
            if self.iterable_dataset or not self.train_test_split:
                train_loader = DataLoader(train_dataset_all, batch_size=self.batch_size,
                                          shuffle=True, num_workers=os.cpu_count())
                return train_loader, None
            else:
                train_size = int(self.train_test_split * len(train_dataset_all))
                val_size = len(train_dataset_all) - train_size
                train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size],
                                                          generator=torch.Generator().manual_seed(42))

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=os.cpu_count())
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                        shuffle=False, num_workers=os.cpu_count())
                return train_loader, val_loader
        else:
            raise Exception('Empty Dataset object given')

    def _prepare_test_parameters(self, experiment_category: str = None, device: str = None, train_house: int = None,
                                 train_set: str = None, time_folds: list = None, fold: int = None):
        if self.experiment_type == SupportedNilmExperiments.CROSS_VALIDATION:
            data = {TEST_HOUSE: [str(train_house)], TEST_SET: [train_set],
                    TEST_DATE: [time_folds[fold][TEST_DATES]]}
        else:
            test_houses = []
            test_sets = []
            test_dates = []
            test_file = open('{}base{}TestSetsInfo_{}'.format(self.test_file_dir, experiment_category, device), 'r')
            for line in test_file:
                toks = line.split(',')
                test_sets.append(toks[0])
                test_houses.append(toks[1])
                test_dates.append([str(toks[2]), str(toks[3].rstrip("\n"))])
            test_file.close()
            data = {TEST_HOUSE: test_houses, TEST_SET: test_sets, TEST_DATE: test_dates}
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    def _prepare_train_eval_input(self, experiment_category: str = None, device: str = None, window: int = None,
                                  model_name: str = None, iteration: int = None, fold: int = None):
        if self.experiment_type == SupportedNilmExperiments.CROSS_VALIDATION:
            datasource, time_folds, train_set, train_house = self._prepare_cv_parameters(experiment_category, device)
            train_dataset_all = self._prepare_cv_dataset(device, fold, window, datasource,
                                                         time_folds, train_set, train_house)
            tests_params = self._prepare_test_parameters(experiment_category, device, train_house,
                                                         train_set, time_folds, fold)
            iteration, train_set_name = fold, train_set
        else:
            train_dataset_all = self._prepare_train_dataset(experiment_category, device, window)
            tests_params = self._prepare_test_parameters(experiment_category, device)
            train_set_name = train_dataset_all.datasource.get_name()
        train_loader, val_loader = self._prepare_train_val_loaders(train_dataset_all)
        mmax, means, stds, meter_means, meter_stds = self.get_dataset_mmax_means_stds(train_dataset_all)

        eval_params = {COLUMN_DEVICE: device,
                       COLUMN_MMAX: mmax,
                       COLUMN_MEANS: meter_means,
                       COLUMN_STDS: meter_stds,
                       COLUMN_GROUNDTRUTH: ''}

        experiment_name = '_'.join([device, experiment_category, TRAIN_NAME, train_set_name, '', ])

        train_eval_args = {
            'model_name': model_name,
            'device': device,
            'window_size': window,
            'exp_type': experiment_category,
            'sample_period': self.sample_period,
            'batch_size': self.batch_size,
            'iteration': iteration,
            'root_dir': self.project_name,
            'model_hparams': self.model_hparams[model_name],
            'save_timeseries': self.save_timeseries,
            'epochs': self.epochs,
            'callbacks': [TrainerCallbacksFactory.create_earlystopping()],
            'train_loader': train_loader,
            'val_loader': val_loader,
            'mmax': mmax,
            'means': means,
            'stds': stds,
            'meter_means': meter_means,
            'meter_stds': meter_stds,
            'tests_params': tests_params,
            'eval_params': eval_params,
            'experiment_name': experiment_name,
        }

        return train_eval_args

    @staticmethod
    def _call_train_eval(args):
        train_eval(**args)

    @staticmethod
    def get_dataset_mmax_means_stds(dataset: Union[ElectricityDataset,
                                                   ElectricityMultiBuildingsDataset,
                                                   ElectricityIterableDataset] = None):
        if dataset:
            mmax = dataset.mmax
            means = dataset.means
            stds = dataset.stds
            meter_means = dataset.meter_means
            meter_stds = dataset.meter_stds
            return mmax, means, stds, meter_means, meter_stds
        else:
            raise Exception('Empty Dataset object given')

    def export_report(self, save_name=STAT_REPORT, stat_measures: list = None):

        report = get_final_report(self.tree_levels, save=True, root_dir=self.project_name, save_name=save_name)
        get_statistical_report(save_name=save_name, data=report, data_filename=None,
                               root_dir=self.project_name, stat_measures=stat_measures)

    def run_experiment(self):

        if self.experiment_type in self.experiments.keys():
            self.experiments[self.experiment_type]()
        else:
            raise Exception('Not supported experiment with name: {}'.format(self.experiment_type))

    def run_benchmark(self):
        for experiment_category in self.experiment_categories:
            for model_name in self.models:
                for device in self.devices:
                    if self.fixed_window:
                        window = self.fixed_window
                    else:
                        if model_name in WINDOWS:
                            window = WINDOWS[model_name][device]
                        else:
                            raise Exception('Model with name {} has not window specified'.format(model_name))
                    if WINDOW_SIZE in self.model_hparams[model_name]:
                        self.model_hparams[model_name][WINDOW_SIZE] = window

                    for iteration in range(1, self.iterations + 1):
                        print('#' * 20)
                        print(ITERATION_NAME, ': ', iteration)
                        print('#' * 20)
                        train_eval_args = self._prepare_train_eval_input(experiment_category, device.value, window,
                                                                         model_name, iteration)
                        self._call_train_eval(
                            train_eval_args
                        )

    def run_cross_validation(self):
        for experiment_category in self.experiment_categories:
            for model_name in self.models:
                for device in self.devices:
                    if self.fixed_window:
                        window = self.fixed_window
                    else:
                        if model_name in WINDOWS:
                            window = WINDOWS[model_name][device]
                        else:
                            raise Exception('Model with name {} has not window specified'.format(model_name))

                    if WINDOW_SIZE in self.model_hparams[model_name]:
                        self.model_hparams[model_name][WINDOW_SIZE] = window

                    for fold in range(self.cv_folds):
                        print('#' * 20)
                        print(FOLD_NAME, ': ', fold)
                        print('#' * 20)
                        train_eval_args = self._prepare_train_eval_input(experiment_category, device.value, window,
                                                                         model_name, None, fold)
                        self._call_train_eval(
                            train_eval_args
                        )

    def run_hyperparameter_tuning_cross_validation(self):
        pass
