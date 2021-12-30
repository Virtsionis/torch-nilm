from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import ElectricalAppliances, SupportedExperimentCategories, SupportedExperimentVolumes

experiment_parameters = {
    EPOCHS: 1,
    ITERATIONS: 1,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    ROLLING_WINDOW: True,
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
        'model_name': 'SimpleGru',
        'hparams': {},
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

model_hparams = ModelHyperModelParameters(model_hparams)
hparam_tuning = HyperParameterTuning(hparam_tuning)

experiment = NILMExperiments(project_name='API_TEST_OBJECTS', clean_project=True,
                             devices=devices, save_timeseries_results=False, experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters, )

experiment.run_benchmark(model_hparams=model_hparams)
experiment.run_cross_validation(model_hparams=model_hparams)
experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
