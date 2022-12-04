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
    FIXED_WINDOW: 200,
    FILLNA_METHOD: SupportedFillingMethods.FILL_ZEROS,
    SUBSEQ_WINDOW: None,
    TRAIN_TEST_SPLIT: 0.75,
    CV_FOLDS: 3,
    NOISE_FACTOR: 0.0,
}


devs_taus = {
    ElectricalAppliances.KETTLE: 0.7, #0.5
    ElectricalAppliances.MICROWAVE: 0.975,
    ElectricalAppliances.FRIDGE: 0.9,
    ElectricalAppliances.WASHING_MACHINE: 0.5, #0.025,
    ElectricalAppliances.DISH_WASHER: 0.5, #0.1,

    # ElectricalAppliances.OVEN: 0.5,
    # ElectricalAppliances.TUMBLE_DRYER: 0.5,
}
devices = list(devs_taus.keys())

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]
prior_distributions = [NORMAL_DIST for i in range(0, len(devices))]
prior_means = [0 for i in range(0, len(devices))]
prior_stds = [.1 for i in range(0, len(devices))]
prior_noise_std = 1

for i, dev in enumerate(devices):
    if dev == ElectricalAppliances.DISH_WASHER:
        prior_stds[i] = 0.15# 0.1 sto 15 kai paei kala
    if dev == ElectricalAppliances.WASHING_MACHINE:
        prior_stds[i] = 0.15#0.1 to krataw
    elif dev == ElectricalAppliances.FRIDGE:
        prior_stds[i] = 0.001# 0.001 to krataw
    elif dev == ElectricalAppliances.KETTLE:
        prior_stds[i] = 0.1# to krataw
    elif dev == ElectricalAppliances.MICROWAVE:
        prior_stds[i] = 0.01#0.1# 0.001 sto 15 kai paei kala

prior_noise_std = 1 - sum(prior_stds)
model_hparams = [

    {
        'model_name': 'UNetNiLM',
        'hparams': {'window_size': None, 'taus': list(devs_taus.values()), 'num_layers': 5, 'features_start': 8,
                    'n_channels': 1, 'num_classes': len(devs_taus.keys()), 'pooling_size': 16,
                    'num_quantiles': 1, 'dropout': 0.1, 'd_model': 128, 'lr': 0.001,
                    'adam_betas': (0.9, 0.98),
                    }
    },
    {
        'model_name': 'CNN1DUnetNilm',
        'hparams': {'window_size': None, 'taus': list(devs_taus.values()), 'num_classes': len(devs_taus.keys()),
                    'pooling_size': 16, 'num_quantiles': 1, 'dropout': 0.0, 'lr': 0.001,
                    'adam_betas': (0.9, 0.98),
                    }
    },

    {
        'model_name': 'VariationalMultiRegressorConvEncoder',
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'beta': 1e-1, 'gamma': 1e-0, 'complexity_cost_weight': 1e-6,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
                    'prior_means': prior_means, 'prior_distributions': prior_distributions,
                    'lr': 1e-3, 'bayesian_encoder': False, 'bayesian_regressor': False,
                    },
    },
    # {
    #     'model_name': 'MultiRegressorConvEncoder',
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'complexity_cost_weight': 1e-6, 'bayesian_encoder': False, 'bayesian_regressor': False,
    #                 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 },
    # },
    # {
    #     'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
    #     # a) deeper shallow nets, b) got rid of reshape layers
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-0,
    #                 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
    #                 'prior_means': prior_means, 'prior_distributions': prior_distributions,
    #                 'lr': 1e-3
    #                 },
    # },
    # {
    #     'model_name': 'SuperEncoder',  # FROM LATENT SPACE but with 2 changes
    #     # a) deeper shallow nets, b) got rid of reshape layers
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-1, 'gamma': 1e-0,
    #                 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
    #                 'prior_means': prior_means, 'prior_distributions': prior_distributions,
    #                 'lr': 1e-3,
    #                 },
    # },
]


model_hparams = ModelHyperModelParameters(model_hparams)
experiment_parameters = ExperimentParameters(**experiment_parameters)

experiment = NILMSuperExperiments(project_name='MULTI_TARGET_FINAL_BENCHMARK', clean_project=False,
                                  devices=devices, save_timeseries_results=False,
                                  experiment_categories=experiment_categories,
                                  experiment_volume=SupportedExperimentVolumes.COMMON_VOLUME,
                                  experiment_parameters=experiment_parameters,
                                  save_model=False, export_plots=False,
                                  )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
#
