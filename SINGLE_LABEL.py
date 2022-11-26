from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
experiment_parameters = {
    EPOCHS: 100,
    ITERATIONS: 3,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    SCALING_METHOD: SupportedScalingMethods.STANDARDIZATION,
    FIXED_WINDOW: 200,
    FILLNA_METHOD: SupportedFillingMethods.FILL_ZEROS,
    SUBSEQ_WINDOW: None,
    TRAIN_TEST_SPLIT: 0.75,
    CV_FOLDS: 3,
    NOISE_FACTOR: 0.0,
}

devices = [
    ElectricalAppliances.KETTLE,
    ElectricalAppliances.MICROWAVE,
    ElectricalAppliances.FRIDGE,
    ElectricalAppliances.WASHING_MACHINE,
    ElectricalAppliances.DISH_WASHER,
    # ElectricalAppliances.OVEN,
    # ElectricalAppliances.TUMBLE_DRYER,
]

prior_distributions = [NORMAL_DIST for i in range(0, len(devices))]
prior_means = [0 for i in range(0, len(devices))]
prior_stds = [.1 for i in range(0, len(devices))]
prior_noise_std = 1

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]

model_hparams = [
    {
        'model_name': 'S2P',
        'hparams': {'window_size': None, 'dropout': 0},
    },
    {
        'model_name': 'NFED',
        'hparams': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
                    'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
    },
    {
        'model_name': 'DAE',
        'hparams': {'input_dim': None},
    },
    {
        'model_name': 'ConvDAE',
        'hparams': {'input_dim': None, 'latent_dim': 32,},
    },
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
        'model_name': 'VariationalMultiRegressorConvEncoder',
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': 1,
                    'beta': 1e-1, 'gamma': 1e-0, 'complexity_cost_weight': 1e-6,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
                    'prior_means': prior_means, 'prior_distributions': prior_distributions,
                    'lr': 1e-3, 'bayesian_encoder': False, 'bayesian_regressor': False,
                    },
    },
    {
        'model_name': 'MultiRegressorConvEncoder',
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': 1,
                    'complexity_cost_weight': 1e-6, 'bayesian_encoder': False, 'bayesian_regressor': False,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    },
    },
    {
        'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
        # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': 1,
                    'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-0,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
                    'prior_means': prior_means, 'prior_distributions': prior_distributions,
                    'lr': 1e-3
                    },
    },
    #  {
    #     'model_name': 'ConvMultiDAE',
    #     'hparams': {'input_dim': None, 'latent_dim': 16 * (len(devices) + 1), 'targets_num': len(devices),
    #                 'output_dim': experiment_parameters[FIXED_WINDOW],
    #                 },
    # },
    # {
    #     'model_name': 'SuperVAEMulti',  # FROM LATENT SPACE but with 2 changes
    #     # a) deeper shallow nets, b) got rid of reshape layers
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-1, 'gamma': 1e-0,
    #                 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
    #                 'prior_means': prior_means, 'prior_distributions': prior_distributions,
    #                 'lr': 1e-3
    #                 },
    # },
    {
        'model_name': 'SuperEncoder',  # FROM LATENT SPACE but with 2 changes
        # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': 1,
                    'alpha': 1e-2, 'beta': 1e-1, 'gamma': 1e-0,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
                    'prior_means': prior_means, 'prior_distributions': prior_distributions,
                    'lr': 1e-3,
                    },
    },
    # {
    #     'model_name': 'SuperVAE',  # FROM LATENT SPACE
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-5, 'gamma': 1e-1, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
    #                 },
    # },
    # {
    #     'model_name': 'SuperVAE1blight',  # FROM LATENT SPACE but with 2 changes
    #     # a) deeper shallow nets, b) got rid of reshape layers
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-2,
    #                 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
    #                 'prior_means': prior_means, 'prior_distributions': prior_distributions,
    #                 'lr': 1e-3
    #                 },
    # },
]


model_hparams = ModelHyperModelParameters(model_hparams)
experiment_parameters = ExperimentParameters(**experiment_parameters)

experiment = NILMExperiments(project_name='SINGLE_LABEL_BENCHMARK', clean_project=True,
                             devices=devices, save_timeseries_results=False,
                             experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.COMMON_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=False, export_plots=False,
                             )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
#