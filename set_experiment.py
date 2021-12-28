from lab.nilm_experiments import NILMExperiments
from constants.constants import *
from constants.enumerates import ElectricalAppliances, SupportedExperimentCategories, \
    SupportedNilmExperiments, SupportedExperimentVolumes

train_params = {
            EPOCHS: 1,
            ITERATIONS: 1,
            SAMPLE_PERIOD: 6,
            BATCH_SIZE: 512,
            ITERABLE_DATASET: False,
            FIXED_WINDOW: 50,
            TRAIN_TEST_SPLIT: 0.8,
            CV_FOLDS: 2,
        }

devices = [
             ElectricalAppliances.KETTLE,
             ElectricalAppliances.MICROWAVE,
             # ElectricalAppliances.FRIDGE,
             # ElectricalAppliances.WASHING_MACHINE,
            ]

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
    # SupportedExperimentCategories.MULTI_CATEGORY
]

experiment_type = SupportedNilmExperiments.BENCHMARK
# experiment_type = SupportedNilmExperiments.CROSS_VALIDATION

model_hparams = {
    'SimpleGru': {},
    # 'SAED': {'window_size': None},
    # 'WGRU': {'dropout': 0},
    # 'S2P': {'window_size': None, 'dropout': 0},
    # 'FNET': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
    #          'input_dim': None, 'hidden_dim': 256, 'dropout': 0},
}

experiment = NILMExperiments(project_name='API_TEST',
                             clean_project=False,
                             experiment_categories=experiment_categories,
                             devices=devices,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             # data_dir=None,
                             save_timeseries_results=True,
                             inference_cpu=False,
                             # train_file_dir=None,
                             # test_file_dir=None,
                             experiment_type=experiment_type,
                             train_params=train_params,
                             model_hparams=model_hparams
                             )
# experiment.run_experiment()
experiment.export_report()

