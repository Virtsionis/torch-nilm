from lab.nilm_experiments import NILMExperiments
from constants.constants import *
from constants.enumerates import ElectricalAppliances, SupportedExperimentCategories, SupportedExperimentVolumes

train_params = {
            EPOCHS: 1,
            ITERATIONS: 1,
            INFERENCE_CPU: False,
            SAMPLE_PERIOD: 6,
            BATCH_SIZE: 1024,
            ITERABLE_DATASET: False,
            FIXED_WINDOW: 50,
            TRAIN_TEST_SPLIT: 0.8,
            CV_FOLDS: 2,
        }

devices = [
             ElectricalAppliances.KETTLE,
             ElectricalAppliances.MICROWAVE,
             ElectricalAppliances.FRIDGE,
             ElectricalAppliances.WASHING_MACHINE,
            ]

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
    SupportedExperimentCategories.MULTI_CATEGORY
]


model_hparams = {
    'SimpleGru': {},
    'SAED': {'window_size': None},
    'WGRU': {'dropout': 0},
    'S2P': {'window_size': None, 'dropout': 0},
    'FNET': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
             'input_dim': None, 'hidden_dim': 256, 'dropout': 0},
}

hparam_tuning = {
    'FNET': [
             {'depth': 5, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
              'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
             {'depth': 3, 'kernel_size': 5, 'cnn_dim': 128, 'dual_cnn': False,
              'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
            ],
    'SAED': [
                    {'window_size': None, 'bidirectional': False, 'hidden_dim': 128},
                    {'window_size': None, 'bidirectional': False, 'hidden_dim': 128, 'num_heads': 4},
            ],
}

experiment = NILMExperiments(project_name='API_TEST_EXPERIMENTS', clean_project=True,
                             devices=devices, save_timeseries_results=True, experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             train_params=train_params,)

experiment.run_benchmark(model_hparams=model_hparams, experiment_categories=experiment_categories)
experiment.run_cross_validation(model_hparams=model_hparams)
experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)

