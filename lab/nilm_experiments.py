import os
import warnings

import torch
import pandas as pd
from typing import Union
from constants.constants import *
from lab.nilm_trainer import train_eval, train_eval_super
from constants.appliance_windows import WINDOWS
from datasources.datasource import Datasource
from datasources.datasource import DatasourceFactory
from torch.utils.data import DataLoader, random_split
from utils.helpers import create_tree_dir, create_time_folds
from callbacks.callbacks_factories import TrainerCallbacksFactory
from utils.nilm_reporting import get_final_report, get_statistical_report
from constants.enumerates import SupportedNilmExperiments, SupportedExperimentCategories, SupportedExperimentVolumes, \
    ElectricalAppliances, SupportedPreprocessingMethods, SupportedFillingMethods, SupportedScalingMethods
from datasources.torchdataset import ElectricityDataset, ElectricityMultiBuildingsDataset, ElectricityIterableDataset, \
    BaseElectricityMultiDataset, UNETBaseElectricityMultiDataset

with torch.no_grad():
    torch.cuda.empty_cache()


class ExperimentParameters:
    """
    A class in order to standardize the general experiment parameters.

    Args:
        epochs (int): the number of training epochs
        iterations (int):  execution iterations of each experiment
        inference_cpu (bool): controls whether the inference should be executed on cpu or on gpu
        sample_period (int): the sample period of the data
        batch_size (int): the batch size
        iterable_dataset (bool):  whether the train dataset should be iterable or not (check: datasources/torchdataset)
        preprocessing_method (SupportedPreprocessingMethods): the desired preprocessing method
        fillna_method (SupportedFillingMethods): the desired filling NA method
        fixed_window (int): (check: datasources/torchdataset)
        subseq_window (int): (check: datasources/torchdataset)
        train_test_split (float):  for train / validation split
        cv_folds (int): the number of cross validation folds
        noise_factor (float): a factor tο multiply a gaussian noise signal, which will be added to the normalized
            mains timeseries. The noise follows a gaussian distribution (mu=0, sigma=1).
            The final signal is given by : mains = mains + noise_factor * np.random(0, 1)

    Example of use:
        experiment_parameters = {
            EPOCHS: 1,
            ITERATIONS: 1,
            INFERENCE_CPU: False,
            SAMPLE_PERIOD: 6,
            BATCH_SIZE: 1024,
            ITERABLE_DATASET: False,
            PREPROCESSING_METHOD: SupportedPreprocessingMethods.MIDPOINT_WINDOW,
            FIXED_WINDOW: 128,
            SUBSEQ_WINDOW: 50,
            TRAIN_TEST_SPLIT: 0.8,
            CV_FOLDS: 2,
        }
        experiment_parameters = ExperimentParameters(**experiment_parameters)

    """
    def __init__(self, epochs: int = 100, iterations: int = 5, inference_cpu: bool = False,
                 sample_period: int = 6, batch_size: int = 256, iterable_dataset: bool = False,
                 preprocessing_method: SupportedPreprocessingMethods = SupportedPreprocessingMethods.ROLLING_WINDOW,
                 scaling_method: SupportedScalingMethods = SupportedScalingMethods.NORMALIZATION,
                 fillna_method: SupportedFillingMethods = SupportedFillingMethods.FILL_ZEROS,
                 fixed_window: int = None, subseq_window: int = None, train_test_split: float = 0.8, cv_folds: int = 3,
                 noise_factor: float = None, ):

        self.params = {
            EPOCHS: epochs,
            ITERATIONS: iterations,
            INFERENCE_CPU: inference_cpu,
            SAMPLE_PERIOD: sample_period,
            BATCH_SIZE: batch_size,
            ITERABLE_DATASET: iterable_dataset,
            PREPROCESSING_METHOD: preprocessing_method,
            SCALING_METHOD: scaling_method,
            FILLNA_METHOD: fillna_method,
            FIXED_WINDOW: fixed_window,
            SUBSEQ_WINDOW: subseq_window,
            TRAIN_TEST_SPLIT: train_test_split,
            CV_FOLDS: cv_folds,
            NOISE_FACTOR: noise_factor,
        }

    def get_params(self):
        return self.params

    def get_param_names(self):
        return self.params.keys()


class BaseModelParameters:
    """
    A base class in order to standardize the model parameters for each experiment type.

    Args:
        params(list): list containing the parameters of every desired model.
    """
    def __init__(self, params: list = None):
        self.params = params

    def get_length(self):
        if isinstance(self.params, list):
            return len(self.params)
        else:
            raise Warning('Input argument should be a list')

    def get_model_names(self):
        if isinstance(self.params, list):
            return [param[MODEL_NAME] for param in self.params]
        else:
            raise Warning('Input argument should be a list')

    def get_model_params(self, model_name: str = None):
        print('MODEL_NAME: ', model_name)
        if model_name and model_name in self.get_model_names():
            for param in self.params:
                if param[MODEL_NAME] == model_name:
                    return param[COLUMN_HPARAMS]

    def _set_model_output_dim(self, model_name: str = None, output_dim: int = 1):
        if output_dim and output_dim > 1 and model_name and model_name in self.get_model_names():
            for param in self.params:
                if param[MODEL_NAME] == model_name:
                    param[COLUMN_HPARAMS][OUTPUT_DIM] = output_dim


class HyperParameterTuning(BaseModelParameters):
    """
    A class in order to standardize the model parameters for HYPERPARAM_TUNE_CV experiment.
    The list of parameters is given in a json-like format. These objects can be used to run the 'export_report' process
    of a project with the same parameters, if the files & folders exist.

    Args:
        params(list): list containing the parameters of every desired model.

    Example of use:
        hparam_tuning = [
            {
                'model_name': 'FNET',
                'hparams': [
                    {'depth': 5, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
                     'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
                    {'depth': 3, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
                     'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
                ]
            },
            {
                'model_name': 'SAED',
                'hparams': [
                    {'window_size': None, 'bidirectional': False, 'hidden_dim': 128},
                    {'window_size': None, 'bidirectional': False, 'hidden_dim': 128, 'num_heads': 4},
                ]
            },
        ]

        hparam_tuning = HyperParameterTuning(hparam_tuning)
        experiment = NILMExperiments(**args)
        experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
        experiment.export_report(hparam_tuning=hparam_tuning, experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)
    """
    def __init__(self, hparam_tuning: list = None):
        super().__init__(params=hparam_tuning)


class ModelHyperModelParameters(BaseModelParameters):
    """
    A class in order to standardize the model parameters for BENCHMARK and CROSS_VALIDATION experiments.
    The list of parameters is given in a json-like format. These objects can be used to run the 'export_report' process
    of a project with the same parameters, if the files & folders exist.

    Args:
        params(list): list containing the parameters of every desired model.

    Example of use:
        model_hparams = [
        {
            'model_name': 'SimpleGru',
            'hparams': {},
        },
        {
            'model_name': 'SAED',
            'hparams': {'window_size': None},
        },
        {
            'model_name': 'WGRU',
            'hparams': {'dropout': 0},
        },
        {
            'model_name': 'S2P',
            'hparams': {'window_size': None, 'dropout': 0},
        },
    ]

    model_hparams = ModelHyperModelParameters(model_hparams)
    experiment = NILMExperiments(**args)

    experiment.run_benchmark(model_hparams=model_hparams)
    experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
                                    -   and/or   -
    experiment.run_cross_validation(model_hparams=model_hparams)
    experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.CROSS_VALIDATION)
    
    """
    def __init__(self, model_hparams: list = None):
        super().__init__(params=model_hparams)


class NILMExperiments:
    """
    The purpose of this class is to organise the NILM experiments/projects in an API-like way. Thus, each type of
    experiment could be executed easily with the minimum number of commands from the user side. Furthermore, each NILM
    project is  organised in separate folders under the main folder 'output'. The class could be easily extended to
    support a wider range of experiments. Currently, three main types of experiments are supported; BENCHMARK,
    CROSS_VALIDATION, HYPERPARAM_TUNE_CV. A description of each experiment is presented below:
        - BENCHMARK :
        - CROSS_VALIDATION :
        - HYPERPARAM_TUNE_CV :

    Args:
        project_name(str): The name of the project
        clean_project(bool): This flag controls whether the files & folders under the same project_name should be
            deleted or not. If value is False, user can add experiments under the same project_name asynchronously
            Default: False
        experiment_categories(list): This list contains the desired experiment_categories to be executed. The available
            categories can be found in constants/enumerates/SupportedExperimentCategories. When empty list is given,
            experiments are executed for all available categories.
        devices(list): This list contains the desired devices to be investigated. The available devices can be found in
            constants/enumerates/ElectricalAppliances.
        save_timeseries_results(bool): The flag controls whether the model output time series should be exported or not.
            It is useful to export the timeseries to visually inspect the output of the models in respect to the ground-
            truth data. Mind that this file is saved in a csv format and could be a handful of MBs in size.
            Default: True
        save_model(bool): The flag controls whether the model weights should be exported or not.
            Mind that this file is saved in a cpkt format and could be a handful of MBs in size.
            Default: False
        export_plots(bool): The flag controls whether result plots should be exported or not. The results come from the
            final report (xlsx). The plots are saved in a 'png' format
            Default: False
        experiment_volume(SupportedExperimentVolumes): The list of the desired experiment_volume to be used.
            The supported volumes can be found in constants/enumerates/SupportedExperimentVolumes.
            Default: SupportedExperimentVolumes.LARGE_VOLUME
        experiment_type(SupportedNilmExperiments): The list of the desired nilm experiments to be executed.
            The supported nilm experiments can be found in constants/enumerates/SupportedNilmExperiments.
        experiment_parameters(ExperimentParameters): The general experiment parameters-settings to be used.
        model_hparams(ModelHyperModelParameters): the model parameters for BENCHMARK and CROSS_VALIDATION experiments
        hparam_tuning(HyperParameterTuning): the model parameters for HYPERPARAM_TUNE_CV experiment
        data_dir(str): The directory of the data. If None is given, the path in datasources/paths_manager.py is used.
        train_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
        test_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.

    Functionality in a nut-shell:
        After input arguments are initialized, the experiment properties can be used as APIs.
        The main properties that could be used for NILM-research are:
            'run_benchmark': this method executes the benchmarking process described in:
                Symeonidis et al. “A Benchmark Framework to Evaluate Energy Disaggregation Solutions.” EANN (2019).
                DOI:10.1007/978-3-030-20257-6_2
            'run_cross_validation': this method executes a Cross Validation process
            'run_hyperparameter_tuning_cross_validation': this method executes a hyperparameter tuning/search using
                Cross Validation.
            'export_report': this method gathers all the experiment reports in a final report in a xlsx type format.
                Basic inputs are the same experiment settings and the experiment_type.

        Every experiment has three main steps in order to be set and executed:
            - Firstly, the hyperparameters of each model are received.
            - Secondly, the input / output of each model is calculated with the use of the '_calculate_model_window' and
                '_set_model_output_dim' methods. It should be noted that the model input & output are highly dependant
                of the preprocessing method that is chosen.
            - Thirdly, the method '_prepare_train_eval_input' prepares the training & evaluation parameters for each
                experiment.

        After an experiment is concluded, a report (xlsx) can be exported either as an API call or internally with
            the flag 'export_report'.

    Example of use:
        experiment_parameters = {
            EPOCHS: 1,
            ITERATIONS: 1,
            INFERENCE_CPU: False,
            SAMPLE_PERIOD: 6,
            BATCH_SIZE: 1024,
            ITERABLE_DATASET: False,
            PREPROCESSING_METHOD: SupportedPreprocessingMethods.MIDPOINT_WINDOW,
            FIXED_WINDOW: None,
            SUBSEQ_WINDOW: None,
            TRAIN_TEST_SPLIT: 0.8,
            CV_FOLDS: 2,
        }

        devices = [
            ElectricalAppliances.KETTLE,
            ElectricalAppliances.MICROWAVE,
        ]

        experiment_categories = [
            SupportedExperimentCategories.SINGLE_CATEGORY,
            SupportedExperimentCategories.MULTI_CATEGORY,
        ]

        model_hparams = [
            {
                'model_name': 'SAED',
                'hparams': {'window_size': None},
            },
            {
                'model_name': 'WGRU',
                'hparams': {'dropout': 0},
            },
        ]

        model_hparams = ModelHyperModelParameters(model_hparams)
        experiment_parameters = ExperimentParameters(**experiment_parameters)

        experiment = NILMExperiments(project_name='NILM_EXPERIMENTS', clean_project=False,
                                     devices=devices, save_timeseries_results=False, experiment_categories=experiment_categories,
                                     experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                                     experiment_parameters=experiment_parameters,
                                     )
        experiment.run_benchmark(model_hparams=model_hparams)
        experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)

    """
    def __init__(self, project_name: str = None, clean_project: bool = False, experiment_categories: list = None,
                 devices: list = None, save_timeseries_results: bool = True, export_plots: bool = True,
                 experiment_volume: SupportedExperimentVolumes = SupportedExperimentVolumes.LARGE_VOLUME,
                 experiment_type: SupportedNilmExperiments = None, experiment_parameters: ExperimentParameters = None,
                 model_hparams: ModelHyperModelParameters = None, hparam_tuning: HyperParameterTuning = None,
                 data_dir: str = None, train_file_dir: str = None, test_file_dir: str = None, save_model: bool = False,
                 save_preprocessing_params: bool = False,):

        self.project_name = project_name
        self.clean_project = clean_project
        self.save_timeseries = save_timeseries_results
        self.export_plots = export_plots
        self.save_model = save_model
        self.save_preprocessing_params = save_preprocessing_params
        self.model_hparams = model_hparams
        self.hparam_tuning = hparam_tuning
        self.experiment_type = experiment_type
        self.experiment_parameters = experiment_parameters
        self.devices = devices
        self.experiment_categories = experiment_categories
        self.experiment_volume = experiment_volume
        self.data_dir = data_dir
        self.train_file_dir = train_file_dir
        self.test_file_dir = test_file_dir

    def _prepare_project_properties(self, devices: list = None, experiment_parameters: ExperimentParameters = None,
                                    data_dir: str = None, train_file_dir: str = None, test_file_dir: str = None,
                                    experiment_volume: SupportedExperimentVolumes = None,
                                    hparam_tuning: HyperParameterTuning = None,
                                    experiment_categories: list = None, model_hparams: ModelHyperModelParameters = None,
                                    experiment_type: SupportedNilmExperiments = None):

        self.model_hparams = model_hparams
        self.hparam_tuning = hparam_tuning
        self._set_models()

        if devices:
            self._set_devices(devices)
        else:
            self._set_devices(self.devices)

        if experiment_parameters:
            self._set_experiment_parameters(experiment_parameters)
        else:
            self._set_experiment_parameters(self.experiment_parameters)

        self._set_supported_experiments()

        if experiment_type:
            self.experiment_type = experiment_type

        if experiment_categories:
            self._set_experiment_categories(experiment_categories)
        else:
            self._set_experiment_categories(self.experiment_categories)

        if data_dir:
            self._set_data_dir(data_dir)
        else:
            self._set_data_dir(self.data_dir)

        if experiment_volume:
            self._set_experiment_volume(experiment_volume)
        else:
            self._set_experiment_volume(self.experiment_volume)

        if train_file_dir and test_file_dir:
            self._set_train_test_file_dir(train_file_dir, test_file_dir)
        else:
            self._set_train_test_file_dir(self.train_file_dir, self.test_file_dir)

        self._create_project_structure()

    def _set_devices(self, devices: list = None):
        if devices and len(devices):
            self.devices = [device.value if isinstance(device, ElectricalAppliances) else device for device in devices]
        else:
            raise Exception('No electrical devices are defined')

    def _set_models(self):
        if self.model_hparams and isinstance(self.model_hparams, ModelHyperModelParameters)\
                and self.model_hparams.get_length():
            self.models = self.model_hparams.get_model_names()
        elif self.hparam_tuning and isinstance(self.hparam_tuning, HyperParameterTuning)\
                and self.hparam_tuning.get_length():
            self.models = self.hparam_tuning.get_model_names()
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

    def _set_default_experiment_parameters(self):
        self.epochs = 100
        self.iterations = 5
        self.inference_cpu = False
        self.sample_period = 6
        self.batch_size = 256
        self.iterable_dataset = False
        self._set_preprocessing_method(SupportedPreprocessingMethods.ROLLING_WINDOW)
        self._set_scaling_method(SupportedScalingMethods.NORMALIZATION)
        self._set_fillna_method(SupportedFillingMethods.FILL_ZEROS)
        self.fixed_window = 100
        self.subseq_window = None
        self.train_test_split = 0.8
        self.cv_folds = 3
        self.noise_factor = None

    def _set_experiment_parameters(self, experiment_parameters: ExperimentParameters = None):
        if experiment_parameters:
            experiment_parameters = experiment_parameters.get_params()
            self.epochs = experiment_parameters[EPOCHS]
            self.iterations = experiment_parameters[ITERATIONS]
            self.inference_cpu = experiment_parameters[INFERENCE_CPU]
            self._set_preprocessing_method(experiment_parameters[PREPROCESSING_METHOD])
            self._set_scaling_method(experiment_parameters[SCALING_METHOD])
            self._set_fillna_method(experiment_parameters[FILLNA_METHOD])
            self.sample_period = experiment_parameters[SAMPLE_PERIOD]
            self.batch_size = experiment_parameters[BATCH_SIZE]
            self.iterable_dataset = experiment_parameters[ITERABLE_DATASET]
            self.fixed_window = experiment_parameters[FIXED_WINDOW]
            self.subseq_window = experiment_parameters[SUBSEQ_WINDOW]
            self.train_test_split = experiment_parameters[TRAIN_TEST_SPLIT]
            self.cv_folds = experiment_parameters[CV_FOLDS]
            self.noise_factor = experiment_parameters[NOISE_FACTOR]
        else:
            warnings.warn('No experiment parameters are defined. So, default parameters will be used.')
            self._set_default_experiment_parameters()

    def _set_experiment_categories(self, experiment_categories: list = None):
        if experiment_categories:
            temp = []
            for experiment_category in experiment_categories:
                if isinstance(experiment_category, SupportedExperimentCategories):
                    temp.append(experiment_category.value)
                else:
                    temp.append(experiment_category)
            self.experiment_categories = temp
        else:
            warnings.warn('No experiment categories are defined. So, both available categories will be executed.')
            self.experiment_categories = [SupportedExperimentCategories.SINGLE_CATEGORY.value,
                                          SupportedExperimentCategories.MULTI_CATEGORY.value
                                          ]

    def _set_data_dir(self, data_dir: str = None,):
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = DIR_DATASETS

    def _set_experiment_volume(self, experiment_volume: SupportedExperimentVolumes = None):
        if experiment_volume:
            if isinstance(self.experiment_volume, SupportedExperimentVolumes):
                self.experiment_volume = experiment_volume.value
            else:
                self.experiment_volume = experiment_volume
        else:
            warnings.warn('No experiment volume is defined. So, large volume of experiments will be used.')
            self.experiment_volume = SupportedExperimentVolumes.LARGE_VOLUME.value

    def _set_preprocessing_method(self, preprocessing_method: SupportedPreprocessingMethods = None):
        if preprocessing_method and isinstance(preprocessing_method, SupportedPreprocessingMethods):
            self.preprocessing_method = preprocessing_method
        else:
            warnings.warn('Preprocessing method was not properly defined. So, ROLLING_WINDOW is used by default.')
            self.preprocessing_method = SupportedPreprocessingMethods.ROLLING_WINDOW

    def _set_scaling_method(self, scaling_method: SupportedScalingMethods = None):
        if scaling_method and isinstance(scaling_method, SupportedScalingMethods):
            self.normalization_method = scaling_method
        else:
            warnings.warn('Scaling method was not properly defined. So, NORMALIZATION is used by default.')
            self.normalization_method = SupportedScalingMethods.NORMALIZATION

    def _set_fillna_method(self, fillna_method: SupportedFillingMethods = None):
        if fillna_method and isinstance(fillna_method, SupportedFillingMethods):
            self.fillna_method = fillna_method
        else:
            warnings.warn('Filling NA method was not properly defined. So, FILL_ZEROS is used by default.')
            self.fillna_method = SupportedFillingMethods.FILL_ZEROS

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
        clean_project = self.clean_project
        project_name = self.project_name
        if isinstance(self.experiment_type, SupportedNilmExperiments):
            experiment_type = self.experiment_type.value
        else:
            experiment_type = self.experiment_type
        if self.save_model:
            level_1 = [DIR_RESULTS_NAME, DIR_SAVED_MODELS_NAME]
        else:
            level_1 = [DIR_RESULTS_NAME]
        tree_levels = {ROOT_LEVEL: project_name,
                       EXPERIMENTS_LEVEL: [experiment_type],
                       LEVEL_1_NAME: level_1,
                       LEVEL_2_NAME: devices,
                       LEVEL_3_NAME: self.models,
                       EXPERIMENTS_NAME: experiment_categories,
                       }
        create_tree_dir(tree_levels=tree_levels, clean=clean_project, plots=self.export_plots)
        self.tree_levels = tree_levels
        self.clean_project = False

    def _prepare_cv_parameters(self, experiment_category: SupportedExperimentVolumes = None, device: str = None):
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

    def _prepare_cv_dataset(self, device: str = None, fold: int = None, window: int = None, datasource: Datasource = None,
                            time_folds: list = None, train_house: int = None):

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
                                                             sample_period=self.sample_period,
                                                             preprocessing_method=self.preprocessing_method,
                                                             normalization_method=self.normalization_method,
                                                             fillna_method=self.fillna_method,
                                                             subseq_window=self.subseq_window,)
        return train_dataset_all

    def _prepare_train_dataset(self, experiment_category: SupportedExperimentCategories = None, device: str = None,
                               window: int = None):
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
                                                                   sample_period=self.sample_period,
                                                                   preprocessing_method=self.preprocessing_method,
                                                                   normalization_method=self.normalization_method,
                                                                   fillna_method=self.fillna_method,
                                                                   subseq_window=self.subseq_window,
                                                                   noise_factor=self.noise_factor)
                else:
                    train_dataset_all = ElectricityDataset(datasource=datasource,
                                                           building=int(train_house),
                                                           window_size=window,
                                                           device=device,
                                                           dates=train_dates,
                                                           sample_period=self.sample_period,
                                                           preprocessing_method=self.preprocessing_method,
                                                           normalization_method=self.normalization_method,
                                                           fillna_method=self.fillna_method,
                                                           subseq_window=self.subseq_window,
                                                           noise_factor=self.noise_factor)

                return train_dataset_all
        file.close()
        train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
                                                             window_size=window,
                                                             sample_period=self.sample_period,
                                                             preprocessing_method=self.preprocessing_method,
                                                             normalization_method=self.normalization_method,
                                                             fillna_method=self.fillna_method,
                                                             subseq_window=self.subseq_window,
                                                             noise_factor=self.noise_factor)
        return train_dataset_all

    def _prepare_train_val_loaders(self, train_dataset_all: Union[ElectricityDataset,
                                                                  BaseElectricityMultiDataset,
                                                                  UNETBaseElectricityMultiDataset,
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
                                  model_name: str = None, iteration: int = None, fold: int = None,
                                  model_hparams: dict = None, model_index: int = None):
        if self.experiment_type in [SupportedNilmExperiments.CROSS_VALIDATION,
                                    SupportedNilmExperiments.HYPERPARAM_TUNE_CV]:
            datasource, time_folds, train_set, train_house = self._prepare_cv_parameters(experiment_category, device)
            train_dataset_all = self._prepare_cv_dataset(device, fold, window, datasource,
                                                         time_folds, train_house)
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
            MODEL_NAME: model_name,
            MODEL_INDEX: model_index,
            COLUMN_DEVICE: device,
            WINDOW_SIZE: window,
            SUBSEQ_WINDOW: self.subseq_window,
            EXPERIMENT_CATEGORY: experiment_category,
            EXPERIMENT_TYPE: self.experiment_type.value,
            SAMPLE_PERIOD: self.sample_period,
            BATCH_SIZE: self.batch_size,
            ITERATION: iteration,
            PREPROCESSING_METHOD: self.preprocessing_method,
            NORMALIZATION_METHOD: self.normalization_method,
            FILLNA_METHOD: self.fillna_method,
            INFERENCE_CPU: self.inference_cpu,
            ROOT_DIR: self.project_name,
            MODE_HPARAMS: model_hparams,
            SAVE_TIMESERIES: self.save_timeseries,
            SAVE_MODEL: self.save_model,
            SAVE_PREPROCESSING_PARAMS: self.save_preprocessing_params,
            EPOCHS: self.epochs,
            CALLBACKS: [TrainerCallbacksFactory.create_earlystopping()],
            TRAIN_LOADER: train_loader,
            VAL_LOADER: val_loader,
            COLUMN_MMAX: mmax,
            COLUMN_MEANS: means,
            COLUMN_STDS: stds,
            METER_MEANS: meter_means,
            METER_STDS: meter_stds,
            TESTS_PARAMS: tests_params,
            EVAL_PARAMS: eval_params,
            EXPERIMENT_NAME: experiment_name,
        }

        return train_eval_args

    @staticmethod
    def _call_train_eval(args):
        train_eval(**args)

    @staticmethod
    def get_dataset_mmax_means_stds(dataset: Union[ElectricityDataset,
                                                   ElectricityMultiBuildingsDataset,
                                                   UNETBaseElectricityMultiDataset,
                                                   BaseElectricityMultiDataset,
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

    def export_report(self, save_name=STAT_REPORT, stat_measures: list = None, devices: list = None,
                      experiment_parameters: list = None, data_dir: str = None, train_file_dir: str = None,
                      test_file_dir: str = None, model_hparams: ModelHyperModelParameters = None,
                      hparam_tuning: HyperParameterTuning = None, experiment_categories: list = None,
                      experiment_volume: SupportedExperimentVolumes = None, experiment_type: SupportedNilmExperiments = None,
                      prepare_project_properties: bool = True, model_index: int = None,
                      ):
        if prepare_project_properties:
            self._prepare_project_properties(devices=devices,
                                             experiment_parameters=experiment_parameters,
                                             data_dir=data_dir,
                                             train_file_dir=train_file_dir,
                                             test_file_dir=test_file_dir,
                                             model_hparams=model_hparams,
                                             hparam_tuning=hparam_tuning,
                                             experiment_volume=experiment_volume,
                                             experiment_categories=experiment_categories,
                                             experiment_type=experiment_type,
                                             )

        if EXPERIMENTS_LEVEL in self.tree_levels and self.tree_levels[EXPERIMENTS_LEVEL] \
                and isinstance(self.tree_levels[EXPERIMENTS_LEVEL], list) and len(self.tree_levels[EXPERIMENTS_LEVEL]):
            root_dir = '/'.join([self.project_name, self.tree_levels[EXPERIMENTS_LEVEL][0], ''])
        else:
            root_dir = self.project_name

        report = get_final_report(self.tree_levels, save=True, root_dir=root_dir, save_name=save_name,
                                  model_index=model_index)
        get_statistical_report(save_name=save_name,
                               data=report,
                               root_dir=root_dir,
                               stat_measures=stat_measures,
                               save_plots=self.export_plots)

    def _set_model_output_dim(self, model_hparams: dict = None, output_dim: int = 1):
        """
        A method that sets the output dimension of each model automatically, based on the preprocessing method that was
            determined by the user.
        """
        if self.preprocessing_method == SupportedPreprocessingMethods.ROLLING_WINDOW:
            model_hparams[OUTPUT_DIM] = 1
        elif self.preprocessing_method == SupportedPreprocessingMethods.MIDPOINT_WINDOW:
            model_hparams[OUTPUT_DIM] = 1
        elif self.preprocessing_method == SupportedPreprocessingMethods.SEQ_T0_SEQ:
            model_hparams[OUTPUT_DIM] = output_dim
        elif self.preprocessing_method == SupportedPreprocessingMethods.SEQ_T0_SUBSEQ:
            if self.subseq_window and self.subseq_window < output_dim:
                model_hparams[OUTPUT_DIM] = self.subseq_window
            else:
                warnings.warn('Sequence window is smaller than subsequence window. So, SEQ_TO_SEQ preprocessing ' +
                              'was applied instead of SEQ_T0_SUBSEQ')
                model_hparams[OUTPUT_DIM] = output_dim
                self.subseq_window = output_dim
                self.preprocessing_method = SupportedPreprocessingMethods.SEQ_T0_SEQ

        return model_hparams

    def _calculate_model_window(self, model_hparams: dict = None, model_name: str = None, device: str = None,):
        """
        A method that sets the input window / input dimension of each model automatically, based on the preprocessing
            method that was determined by the user. Bellow, some use-case scenarios are explored:
            - If user chose FIXED_WINDOW, the input of the models are set to that value.
            - If user chose FIXED_WINDOW=None, the input of the models are set based on the predefined windows that can
                be found in constants/appliance_windows.py.
            - If user chose FIXED_WINDOW=None and specifically set window_size/input_dim to a value, that value is taken
                into account.
        """

        if WINDOW_SIZE in model_hparams and model_hparams[WINDOW_SIZE]:
            window = model_hparams[WINDOW_SIZE]
        elif INPUT_DIM in model_hparams and model_hparams[INPUT_DIM]:
            window = model_hparams[INPUT_DIM]
        else:
            if self.fixed_window:
                window = self.fixed_window
            else:
                if model_name in WINDOWS:
                    dev = ElectricalAppliances(device)
                    window = WINDOWS[model_name][dev]
                else:
                    raise Exception('Model with name {} has not window specified'.format(model_name))
            if WINDOW_SIZE in model_hparams:
                model_hparams[WINDOW_SIZE] = window
            elif INPUT_DIM in model_hparams:
                model_hparams[INPUT_DIM] = window
            if WINDOW_SIZE in model_hparams:
                model_hparams[WINDOW_SIZE] = window
            elif INPUT_DIM in model_hparams:
                model_hparams[INPUT_DIM] = window
        return model_hparams, window

    def run_benchmark(self, devices: list = None, experiment_parameters: list = None, data_dir: str = None,
                      train_file_dir: str = None, test_file_dir: str = None, model_hparams: ModelHyperModelParameters = None,
                      experiment_volume: SupportedExperimentVolumes = None, experiment_categories: list = None,
                      export_report: bool = True, stat_measures: list = None, ):
        """
        A method to execute the benchmark methodology described in:
            Symeonidis et al. “A Benchmark Framework to Evaluate Energy Disaggregation Solutions.” EANN (2019).
            DOI:10.1007/978-3-030-20257-6_2

        Args:
            devices(list): This list contains the desired devices to be investigated. The available devices can be found in
                constants/enumerates/ElectricalAppliances.
            experiment_parameters(list): The general experiment parameters-settings to be used.
            data_dir(str): The directory of the data. If None is given, the path in datasources/paths_manager.py is used
            train_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
            test_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
            model_hparams(ModelHyperModelParameters): The hyperparameters for all the models under investigation.
            experiment_volume(SupportedExperimentVolumes): The list of the desired experiment_volume to be used.
            experiment_categories(list): This list contains the desired experiment_categories to be executed.
                The available categories can be found in constants/enumerates/SupportedExperimentCategories.
            export_report(bool): Whether to export the final report (xlsx) or not.
            stat_measures(list): user can define the appropriate statistical measures to be included to the report
                supported measures: [ MEAN, MEDIAN, STANDARD_DEVIATION, MINIMUM, MAXIMUM, PERCENTILE_25TH,
                PERCENTILE_75TH]
        Example of use:
            model_hparams = [
                {
                    'model_name': 'SAED',
                    'hparams': {'window_size': None},
                },
                {
                    'model_name': 'WGRU',
                    'hparams': {'dropout': 0},
                },
            ]
            model_hparams = ModelHyperModelParameters(model_hparams)
            experiment_parameters = ExperimentParameters(**experiment_parameters)

            experiment = NILMExperiments(project_name='NILM_EXPERIMENTS', clean_project=False,
                                         devices=devices, save_timeseries_results=False,
                                         experiment_categories=experiment_categories,
                                         experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                                         experiment_parameters=experiment_parameters,
                                         )
            experiment.run_benchmark(model_hparams=model_hparams)
        """
        print('>>>BENCHMARK EXPERIMENT<<<')
        self._prepare_project_properties(devices=devices,
                                         experiment_parameters=experiment_parameters,
                                         data_dir=data_dir,
                                         train_file_dir=train_file_dir,
                                         test_file_dir=test_file_dir,
                                         experiment_volume=experiment_volume,
                                         model_hparams=model_hparams,
                                         hparam_tuning=None,
                                         experiment_categories=experiment_categories,
                                         experiment_type=SupportedNilmExperiments.BENCHMARK,
                                         )

        for experiment_category in self.experiment_categories:
            print('EXPERIMENT CATEGORY: ', experiment_category)
            for model_name in self.models:
                model_hparams = self.model_hparams.get_model_params(model_name)
                for device in self.devices:
                    model_hparams, window = self._calculate_model_window(model_hparams=model_hparams,
                                                                         model_name=model_name, device=device)
                    model_hparams = self._set_model_output_dim(model_hparams, output_dim=window)

                    for iteration in range(1, self.iterations + 1):
                        print('#' * 20)
                        print(ITERATION_NAME, ': ', iteration)
                        print('#' * 20)
                        train_eval_args = self._prepare_train_eval_input(experiment_category, device, window,
                                                                         model_name, iteration, None,
                                                                         model_hparams=model_hparams)
                        self._call_train_eval(
                            train_eval_args
                        )
        if export_report:
            self.export_report(save_name=STAT_REPORT,
                               stat_measures=stat_measures,
                               prepare_project_properties=False,
                               )

    def run_cross_validation(self, devices: list = None, experiment_parameters: ExperimentParameters = None,
                             data_dir: str = None, train_file_dir: str = None, test_file_dir: str = None,
                             model_hparams: ModelHyperModelParameters = None,
                             experiment_volume: SupportedExperimentVolumes = None, experiment_categories: list = None,
                             export_report: bool = True, stat_measures: list = None, ):
        """
         A method to execute a cross validation method.

         Args:
             devices(list): This list contains the desired devices to be investigated. The available devices can be found in
                 constants/enumerates/ElectricalAppliances.
             experiment_parameters(list): The general experiment parameters-settings to be used.
             data_dir(str): The directory of the data. If None is given, the path in datasources/paths_manager.py is used
             train_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
             test_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
             model_hparams(ModelHyperModelParameters): The hyperparameters for all the models under investigation.
             experiment_volume(SupportedExperimentVolumes): The list of the desired experiment_volume to be used.
             experiment_categories(list): This list contains the desired experiment_categories to be executed.
                 The available categories can be found in constants/enumerates/SupportedExperimentCategories.
             export_report(bool): Whether to export the final report (xlsx) or not.
             stat_measures(list): user can define the appropriate statistical measures to be included to the report
                 supported measures: [ MEAN, MEDIAN, STANDARD_DEVIATION, MINIMUM, MAXIMUM, PERCENTILE_25TH,
                 PERCENTILE_75TH]
         Example of use:
             model_hparams = [
                 {
                     'model_name': 'SAED',
                     'hparams': {'window_size': None},
                 },
                 {
                     'model_name': 'WGRU',
                     'hparams': {'dropout': 0},
                 },
             ]
             model_hparams = ModelHyperModelParameters(model_hparams)
             experiment_parameters = ExperimentParameters(**experiment_parameters)

             experiment = NILMExperiments(project_name='NILM_EXPERIMENTS', clean_project=False,
                                          devices=devices, save_timeseries_results=False,
                                          experiment_categories=experiment_categories,
                                          experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                                          experiment_parameters=experiment_parameters,
                                          )
             experiment.run_cross_validation(model_hparams=model_hparams)
        """
        print('>>>CROSS VALIDATION EXPERIMENT<<<')
        self._prepare_project_properties(devices=devices,
                                         experiment_parameters=experiment_parameters,
                                         data_dir=data_dir,
                                         train_file_dir=train_file_dir,
                                         test_file_dir=test_file_dir,
                                         experiment_volume=experiment_volume,
                                         model_hparams=model_hparams,
                                         hparam_tuning=None,
                                         experiment_categories=experiment_categories,
                                         experiment_type=SupportedNilmExperiments.CROSS_VALIDATION,
                                         )
        for experiment_category in self.experiment_categories:
            print('EXPERIMENT CATEGORY: ', experiment_category)
            for model_name in self.models:
                model_hparams = self.model_hparams.get_model_params(model_name)
                for device in self.devices:
                    model_hparams, window = self._calculate_model_window(model_hparams=model_hparams,
                                                                         model_name=model_name, device=device)
                    model_hparams = self._set_model_output_dim(model_hparams, output_dim=window)

                    for fold in range(self.cv_folds):
                        print('#' * 20)
                        print(FOLD_NAME, ': ', fold)
                        print('#' * 20)
                        train_eval_args = self._prepare_train_eval_input(experiment_category, device, window,
                                                                         model_name, None, fold,
                                                                         model_hparams=model_hparams)
                        self._call_train_eval(
                            train_eval_args
                        )
        if export_report:
            self.export_report(save_name=STAT_REPORT,
                               stat_measures=stat_measures,
                               prepare_project_properties=False,
                               )

    def run_hyperparameter_tuning_cross_validation(self, devices: list = None, experiment_parameters: list = None,
                                                   data_dir: str = None, train_file_dir: str = None,
                                                   test_file_dir: str = None,
                                                   experiment_volume: SupportedExperimentVolumes = None,
                                                   hparam_tuning: HyperParameterTuning = None,
                                                   experiment_categories: list = None,
                                                   export_report: bool = True, stat_measures: list = None, ):
        """
         A method to execute hyperparameter tuning using a cross validation method.

         Args:
             devices(list): This list contains the desired devices to be investigated. The available devices can be found in
                 constants/enumerates/ElectricalAppliances.
             experiment_parameters(list): The general experiment parameters-settings to be used.
             data_dir(str): The directory of the data. If None is given, the path in datasources/paths_manager.py is used
             train_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
             test_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
             hparam_tuning(HyperParameterTuning): The hyperparameters for all the models under investigation.
             experiment_volume(SupportedExperimentVolumes): The list of the desired experiment_volume to be used.
             experiment_categories(list): This list contains the desired experiment_categories to be executed.
                 The available categories can be found in constants/enumerates/SupportedExperimentCategories.
             export_report(bool): Whether to export the final report (xlsx) or not.
             stat_measures(list): user can define the appropriate statistical measures to be included to the report
                 supported measures: [ MEAN, MEDIAN, STANDARD_DEVIATION, MINIMUM, MAXIMUM, PERCENTILE_25TH,
                 PERCENTILE_75TH]
         Example of use:
             hparam_tuning = [
                {
                    'model_name': 'FNET',
                    'hparams': [
                        {'depth': 5, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
                         'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
                        {'depth': 3, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
                         'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
                    ]
                },
                {
                    'model_name': 'SAED',
                    'hparams': [
                        {'window_size': None, 'bidirectional': False, 'hidden_dim': 128},
                        {'window_size': None, 'bidirectional': False, 'hidden_dim': 128, 'num_heads': 4},
                    ]
                },
            ]
             hparam_tuning = HyperParameterTuning(hparam_tuning)
             experiment_parameters = ExperimentParameters(**experiment_parameters)

             experiment = NILMExperiments(project_name='NILM_EXPERIMENTS', clean_project=False,
                                          devices=devices, save_timeseries_results=False,
                                          experiment_categories=experiment_categories,
                                          experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                                          experiment_parameters=experiment_parameters,
                                          )
             experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
        """
        print('>>>HYPERPARAMETER TUNING EXPERIMENT<<<')
        self._prepare_project_properties(devices=devices,
                                         experiment_parameters=experiment_parameters,
                                         data_dir=data_dir,
                                         train_file_dir=train_file_dir,
                                         test_file_dir=test_file_dir,
                                         experiment_volume=experiment_volume,
                                         model_hparams=None,
                                         hparam_tuning=hparam_tuning,
                                         experiment_categories=experiment_categories,
                                         experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV,
                                         )

        for experiment_category in self.experiment_categories:
            print('EXPERIMENT CATEGORY: ', experiment_category)
            for model_name in self.models:
                model_hparams_list = self.hparam_tuning.get_model_params(model_name)
                for model_index, model_hparams in enumerate(model_hparams_list):
                    print(model_hparams, model_index)
                    for device in self.devices:
                        model_hparams, window = self._calculate_model_window(model_hparams=model_hparams,
                                                                             model_name=model_name, device=device)
                        model_hparams = self._set_model_output_dim(model_hparams, output_dim=window)

                        for fold in range(self.cv_folds):
                            print('#' * 20)
                            print(FOLD_NAME, ': ', fold)
                            print('#' * 20)
                            train_eval_args = self._prepare_train_eval_input(experiment_category, device, window,
                                                                             model_name, None, fold,
                                                                             model_index=model_index + 1,
                                                                             model_hparams=model_hparams)
                            self._call_train_eval(
                                train_eval_args
                            )
        if export_report:
            self.export_report(save_name=STAT_REPORT,
                               stat_measures=stat_measures,
                               prepare_project_properties=False,
                               model_index=model_index + 1,
                               )


class NILMSuperExperiments(NILMExperiments):
    def __init__(self, project_name: str = None, clean_project: bool = False, experiment_categories: list = None,
                 devices: list = None, save_timeseries_results: bool = True, export_plots: bool = True,
                 experiment_volume: SupportedExperimentVolumes = SupportedExperimentVolumes.LARGE_VOLUME,
                 experiment_type: SupportedNilmExperiments = None, experiment_parameters: ExperimentParameters = None,
                 model_hparams: ModelHyperModelParameters = None, hparam_tuning: HyperParameterTuning = None,
                 data_dir: str = None, train_file_dir: str = None, test_file_dir: str = None, save_model: bool = False,
                 save_preprocessing_params: bool = False):

        super(NILMSuperExperiments, self).__init__(project_name, clean_project, experiment_categories, devices, save_timeseries_results,
                                                   export_plots, experiment_volume, experiment_type, experiment_parameters, model_hparams,
                                                   hparam_tuning, data_dir, train_file_dir, test_file_dir, save_model, save_preprocessing_params)

    def _set_default_experiment_parameters(self):
        self.epochs = 100
        self.iterations = 5
        self.inference_cpu = False
        self.sample_period = 6
        self.batch_size = 256
        self.iterable_dataset = False
        self._set_preprocessing_method(SupportedPreprocessingMethods.SEQ_T0_SEQ)
        self._set_scaling_method(SupportedScalingMethods.NORMALIZATION)
        self._set_fillna_method(SupportedFillingMethods.FILL_ZEROS)
        self.fixed_window = 100
        self.subseq_window = None
        self.train_test_split = 0.8
        self.cv_folds = 3
        self.noise_factor = None

    def run_benchmark(self, devices: list = None, experiment_parameters: list = None, data_dir: str = None,
                      train_file_dir: str = None, test_file_dir: str = None, model_hparams: ModelHyperModelParameters = None,
                      experiment_volume: SupportedExperimentVolumes = None, experiment_categories: list = None,
                      export_report: bool = True, stat_measures: list = None, ):
        """
        A method to execute the benchmark methodology described in:
            Symeonidis et al. “A Benchmark Framework to Evaluate Energy Disaggregation Solutions.” EANN (2019).
            DOI:10.1007/978-3-030-20257-6_2

        Args:
            devices(list): This list contains the desired devices to be investigated. The available devices can be found in
                constants/enumerates/ElectricalAppliances.
            experiment_parameters(list): The general experiment parameters-settings to be used.
            data_dir(str): The directory of the data. If None is given, the path in datasources/paths_manager.py is used
            train_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
            test_file_dir(str): The directory of the date files. If None is given, the files in benchmark dir are used.
            model_hparams(ModelHyperModelParameters): The hyperparameters for all the models under investigation.
            experiment_volume(SupportedExperimentVolumes): The list of the desired experiment_volume to be used.
            experiment_categories(list): This list contains the desired experiment_categories to be executed.
                The available categories can be found in constants/enumerates/SupportedExperimentCategories.
            export_report(bool): Whether to export the final report (xlsx) or not.
            stat_measures(list): user can define the appropriate statistical measures to be included to the report
                supported measures: [ MEAN, MEDIAN, STANDARD_DEVIATION, MINIMUM, MAXIMUM, PERCENTILE_25TH,
                PERCENTILE_75TH]
        Example of use:
            model_hparams = [
                {
                    'model_name': 'SAED',
                    'hparams': {'window_size': None},
                },
                {
                    'model_name': 'WGRU',
                    'hparams': {'dropout': 0},
                },
            ]
            model_hparams = ModelHyperModelParameters(model_hparams)
            experiment_parameters = ExperimentParameters(**experiment_parameters)

            experiment = NILMExperiments(project_name='NILM_EXPERIMENTS', clean_project=False,
                                         devices=devices, save_timeseries_results=False,
                                         experiment_categories=experiment_categories,
                                         experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                                         experiment_parameters=experiment_parameters,
                                         )
            experiment.run_benchmark(model_hparams=model_hparams)
        """
        print('>>>BENCHMARK EXPERIMENT<<<')
        self._prepare_project_properties(devices=devices,
                                         experiment_parameters=experiment_parameters,
                                         data_dir=data_dir,
                                         train_file_dir=train_file_dir,
                                         test_file_dir=test_file_dir,
                                         experiment_volume=experiment_volume,
                                         model_hparams=model_hparams,
                                         hparam_tuning=None,
                                         experiment_categories=experiment_categories,
                                         experiment_type=SupportedNilmExperiments.BENCHMARK,
                                         )

        for experiment_category in self.experiment_categories:
            print('EXPERIMENT CATEGORY: ', experiment_category)
            print(self.models)
            for model_name in self.models:
                model_hparams = self.model_hparams.get_model_params(model_name)
                model_hparams, window = self._calculate_model_window(model_hparams=model_hparams,
                                                                     model_name=model_name, device=self.devices[0])
                model_hparams = self._set_model_output_dim(model_hparams, output_dim=window)

                for iteration in range(1, self.iterations + 1):
                    print('#' * 20)
                    print(ITERATION_NAME, ': ', iteration)
                    print('#' * 20)
                    train_eval_args = self._prepare_train_eval_input(experiment_category, self.devices, window,
                                                                     model_name, iteration, None,
                                                                     model_hparams=model_hparams)
                    self._call_train_eval(
                        train_eval_args
                    )
        if export_report:
            self.export_report(save_name=STAT_REPORT,
                               stat_measures=stat_measures,
                               prepare_project_properties=False,
                               )

    def _prepare_train_dataset(self, experiment_category: SupportedExperimentCategories = None, devices: list = None,
                               window: int = None, model_name: str = None):
        file = open('{}base{}TrainSetsInfo_{}'.format(self.train_file_dir, experiment_category, devices[0]), 'r')
        train_info = []
        for line in file:
            toks = line.split(',')
            train_set = toks[0]
            train_house = toks[1]
            train_dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
            datasource = DatasourceFactory.create_datasource(train_set)
            if experiment_category == SupportedExperimentCategories.MULTI_CATEGORY:
                train_info.append({
                    COLUMN_DEVICE: devices,
                    COLUMN_DATASOURCE: datasource,
                    COLUMN_BUILDING: int(train_house),
                    COLUMN_DATES: train_dates,
                })
            else:
                file.close()
                if self.iterable_dataset:
                    pass
                    # TODO: Iterable version of BaseElectricityMultiDataset
                    # train_dataset_all = ElectricityIterableDataset(datasource=datasource,
                    #                                                building=int(train_house),
                    #                                                window_size=window,
                    #                                                device=device,
                    #                                                dates=train_dates,
                    #                                                sample_period=self.sample_period,
                    #                                                preprocessing_method=self.preprocessing_method,
                    #                                                fillna_method=self.fillna_method,
                    #                                                subseq_window=self.subseq_window,
                    #                                                noise_factor=self.noise_factor)
                else:
                    if (model_name == 'UNetNiLM') or (model_name == 'CNN1DUnetNilm') or \
                            (model_name == 'StateVariationalMultiRegressorConvEncoder') or \
                            (model_name == 'MultiRegressorConvEncoder'):
                        train_dataset_all = UNETBaseElectricityMultiDataset(datasource=datasource,
                                                                            building=int(train_house),
                                                                            window_size=window,
                                                                            devices=devices,
                                                                            start_date=train_dates[0],
                                                                            end_date=train_dates[1],
                                                                            sample_period=self.sample_period,
                                                                            subseq_window=self.subseq_window,
                                                                            noise_factor=self.noise_factor,
                                                                            preprocessing_method=self.preprocessing_method,
                                                                            normalization_method=self.normalization_method)

                    else:
                        train_dataset_all = BaseElectricityMultiDataset(datasource=datasource,
                                                                        building=int(train_house),
                                                                        window_size=window,
                                                                        devices=devices,
                                                                        start_date=train_dates[0],
                                                                        end_date=train_dates[1],
                                                                        sample_period=self.sample_period,
                                                                        subseq_window=self.subseq_window,
                                                                        noise_factor=self.noise_factor,
                                                                        preprocessing_method=self.preprocessing_method,
                                                                        normalization_method=self.normalization_method)
                return train_dataset_all
        file.close()
        # TODO: Multi buildings version for BaseElectricityMultiDataset
        # train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
        #                                                      window_size=window,
        #                                                      sample_period=self.sample_period,
        #                                                      preprocessing_method=self.preprocessing_method,
        #                                                      fillna_method=self.fillna_method,
        #                                                      subseq_window=self.subseq_window,
        #                                                      noise_factor=self.noise_factor)
        # return train_dataset_all

    def _prepare_train_eval_input(self, experiment_category: str = None, devices: list = None, window: int = None,
                                  model_name: str = None, iteration: int = None, fold: int = None,
                                  model_hparams: dict = None, model_index: int = None):

        # TODO: Provide support for the rest of experiments
        # if self.experiment_type in [SupportedNilmExperiments.CROSS_VALIDATION,
        #                             SupportedNilmExperiments.HYPERPARAM_TUNE_CV]:
        #     datasource, time_folds, train_set, train_house = self._prepare_cv_parameters(experiment_category, device)
        #     train_dataset_all = self._prepare_cv_dataset(device, fold, window, datasource,
        #                                                  time_folds, train_house)
        #     tests_params = self._prepare_test_parameters(experiment_category, device, train_house,
        #                                                  train_set, time_folds, fold)
        #     iteration, train_set_name = fold, train_set
        # else:
        train_dataset_all = self._prepare_train_dataset(experiment_category, devices, window, model_name)
        tests_params = self._prepare_test_parameters(experiment_category, devices[0])
        train_set_name = train_dataset_all.datasource.get_name()
        train_loader, val_loader = self._prepare_train_val_loaders(train_dataset_all)
        mmax, means, stds, meter_means, meter_stds = self.get_dataset_mmax_means_stds(train_dataset_all)

        eval_params = {COLUMN_DEVICE: devices,
                       COLUMN_MMAX: mmax,
                       COLUMN_MEANS: meter_means,
                       COLUMN_STDS: meter_stds,
                       COLUMN_GROUNDTRUTH: '', }

        experiment_name = '_'.join([experiment_category, TRAIN_NAME, train_set_name, '', ])

        train_eval_args = {
            MODEL_NAME: model_name,
            MODEL_INDEX: model_index,
            COLUMN_DEVICES: devices,
            WINDOW_SIZE: window,
            SUBSEQ_WINDOW: self.subseq_window,
            EXPERIMENT_CATEGORY: experiment_category,
            EXPERIMENT_TYPE: self.experiment_type.value,
            SAMPLE_PERIOD: self.sample_period,
            BATCH_SIZE: self.batch_size,
            ITERATION: iteration,
            PREPROCESSING_METHOD: self.preprocessing_method,
            NORMALIZATION_METHOD: self.normalization_method,
            FILLNA_METHOD: self.fillna_method,
            INFERENCE_CPU: self.inference_cpu,
            ROOT_DIR: self.project_name,
            MODE_HPARAMS: model_hparams,
            SAVE_TIMESERIES: self.save_timeseries,
            SAVE_MODEL: self.save_model,
            SAVE_PREPROCESSING_PARAMS: self.save_preprocessing_params,
            EPOCHS: self.epochs,
            CALLBACKS: [TrainerCallbacksFactory.create_earlystopping()],
            TRAIN_LOADER: train_loader,
            VAL_LOADER: val_loader,
            COLUMN_MMAX: mmax,
            COLUMN_MEANS: means,
            COLUMN_STDS: stds,
            METER_MEANS: meter_means,
            METER_STDS: meter_stds,
            TESTS_PARAMS: tests_params,
            EVAL_PARAMS: eval_params,
            EXPERIMENT_NAME: experiment_name,
        }

        return train_eval_args

    @staticmethod
    def _call_train_eval(args):
        train_eval_super(**args)
