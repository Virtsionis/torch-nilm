from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
experiment_parameters = {
    EPOCHS: 100,
    ITERATIONS: 10,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    SCALING_METHOD: SupportedScalingMethods.STANDARDIZATION,
    FILLNA_METHOD: SupportedFillingMethods.FILL_INTERPOLATION,
    FIXED_WINDOW: None,
    SUBSEQ_WINDOW: 50,
    TRAIN_TEST_SPLIT: 0.75,
    CV_FOLDS: 3,
    NOISE_FACTOR: 0,
}

devices = [
    ElectricalAppliances.KETTLE,
    ElectricalAppliances.MICROWAVE,
    ElectricalAppliances.FRIDGE,
    ElectricalAppliances.WASHING_MACHINE,
    ElectricalAppliances.DISH_WASHER,
]

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
    SupportedExperimentCategories.MULTI_CATEGORY,
]

model_hparams = [

    {
        'model_name': 'SimpleGru',
        'hparams': {},
    },
    {
        'model_name': 'S2P',
        'hparams': {'window_size': None},
    },
    {
        'model_name': 'DAE',
        'hparams': {'input_dim': None},
    },
    {
        'model_name': 'NFED',
        'hparams': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
                    'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
    },
    {
        'model_name': 'SAED',
        'hparams': {'window_size': None},
    },
    {
        'model_name': 'WGRU',
        'hparams': {'dropout': 0},
    },
]

hparam_tuning = [
    {
        'model_name': 'NFED',
        'hparams': [
            {'depth': 1, 'kernel_size': 5, 'cnn_dim': 16,
             'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
            {'depth': 2, 'kernel_size': 5, 'cnn_dim': 32,
             'input_dim': None, 'hidden_dim': 64, 'dropout': 0.0},
        ]
    },
    {
        'model_name': 'SAED',
        'hparams': [
            {'window_size': None, 'bidirectional': False, 'hidden_dim': 16},
            {'window_size': None, 'bidirectional': False, 'hidden_dim': 16, 'num_heads': 2},
        ]
    },
]

model_hparams = ModelHyperModelParameters(model_hparams)
hparam_tuning = HyperParameterTuning(hparam_tuning)
experiment_parameters = ExperimentParameters(**experiment_parameters)

experiment = NILMExperiments(project_name='test', clean_project=False,
                             devices=devices, save_timeseries_results=True, experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=True, export_plots=True,
                             )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
# experiment.run_cross_validation(model_hparams=model_hparams)
# experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
# experiment.export_report(hparam_tuning=hparam_tuning, experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)