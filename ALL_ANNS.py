from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *

for sampling in [6]:
    experiment_parameters = {
        EPOCHS: 1,
        ITERATIONS: 3,
        BATCH_SIZE: 256,
        INFERENCE_CPU: False,
        SAMPLE_PERIOD: sampling,
        ITERABLE_DATASET: False,
        PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
        FILLNA_METHOD: SupportedFillingMethods.FILL_ZEROS,
        FIXED_WINDOW: 100,
        SUBSEQ_WINDOW: 50,
        TRAIN_TEST_SPLIT: 0.75,
        CV_FOLDS: 3,
        NOISE_FACTOR: 0.05,
    }

    devices = [
        # WaterAppliances.BIDET,
        # WaterAppliances.SHOWER,
        # WaterAppliances.WASHBASIN,
        # WaterAppliances.KITCHENFAUCET,
        WaterAppliances.WASHING_MACHINE_W,
    ]

    experiment_categories = [
        SupportedExperimentCategories.SINGLE_CATEGORY,
    ]

    model_hparams = [


        # {
        #     'model_name': 'DAE',
        #     'hparams': {'input_dim': None},
        # },
        # {
        #     'model_name': 'SimpleGru',
        #     'hparams': {},
        # },
        # {
        #     'model_name': 'SAED',
        #     'hparams': {'window_size': None},
        # },
        # {
        #     'model_name': 'NFED',
        #     'hparams': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
        #                 'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
        # },
    {
        'model_name': 'S2P',
        'hparams': {'window_size': None, 'dropout': 0},
    },
        # {
        #     'model_name': 'WGRU',
        #     'hparams': {'dropout': 0},
        # },
    ]

    model_hparams = ModelHyperModelParameters(model_hparams)
    experiment_parameters = ExperimentParameters(**experiment_parameters)

    experiment = NILMExperiments(project_name='iWet_comparisons_{}2sec'.format(str(sampling)), clean_project=False,
                                 devices=devices, save_timeseries_results=False, experiment_categories=experiment_categories,
                                 experiment_volume=SupportedExperimentVolumes.WATER_VOLUME,
                                 experiment_parameters=experiment_parameters,
                                 save_model=False, export_plots=True,
                                 )

    experiment.run_water_benchmark(model_hparams=model_hparams, transfer_learning=False)
    # experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
    # experiment.run_water_cross_validation(model_hparams=model_hparams)
