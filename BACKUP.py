from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
experiment_parameters = {
    EPOCHS: 1,
    ITERATIONS: 3,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    SCALING_METHOD: SupportedScalingMethods.STANDARDIZATION,
    FIXED_WINDOW: 10,
    FILLNA_METHOD: SupportedFillingMethods.FILL_ZEROS,
    SUBSEQ_WINDOW: None,
    TRAIN_TEST_SPLIT: 0.75,
    CV_FOLDS: 3,
    NOISE_FACTOR: 0.0,
}

devices = [
    ElectricalAppliances.KETTLE,
    # ElectricalAppliances.MICROWAVE,
    # ElectricalAppliances.FRIDGE,
    # ElectricalAppliances.WASHING_MACHINE,
    # ElectricalAppliances.DISH_WASHER,
    # ElectricalAppliances.OVEN,
    # ElectricalAppliances.TUMBLE_DRYER,
]

prior_distributions = [NORMAL_DIST for i in range(0, len(devices))]
prior_means = [0 for i in range(0, len(devices))]
prior_stds = [.1 for i in range(0, len(devices))]
prior_noise_std = 1

# for i, dev in enumerate(devices):
#     if dev == ElectricalAppliances.DISH_WASHER:
#         prior_distributions[i] = LAPLACE_DIST
#         prior_stds[i] = 0.0077#0.15# 0.1 sto 15 kai paei kala
#         prior_means[i] = 0.611
    # elif dev == ElectricalAppliances.WASHING_MACHINE:
    #     prior_distributions[i] = LAPLACE_DIST
    #     prior_stds[i] = 0.071#0.15#0.1 to krataw
    #     prior_means[i] = 0.454
    # elif dev == ElectricalAppliances.FRIDGE:
    #     prior_distributions[i] = STUDENT_T_DIST
    #     prior_stds[i] = 0.022#0.001# 0.001 to krataw
    #     prior_means[i] = 0.1376
    # elif dev == ElectricalAppliances.KETTLE:
    #     prior_distributions[i] = CAUCHY_DIST
    #     prior_stds[i] = 0.0067#0.1# to krataw
    #     prior_means[i] = 0.6794
    # elif dev == ElectricalAppliances.MICROWAVE:
    #     prior_distributions[i] = CAUCHY_DIST
    #     prior_stds[i] = 0.0077#0.001#0.1# 0.001 sto 15 kai paei kala
    #     prior_means[i] = 0.4942


# for i, dev in enumerate(devices):
#     # if dev == ElectricalAppliances.DISH_WASHER:
#     #     prior_stds[i] = 0.15# 0.1 sto 15 kai paei kala
#     if dev == ElectricalAppliances.WASHING_MACHINE:
#         prior_stds[i] = 0.15#0.1 to krataw
#     elif dev == ElectricalAppliances.FRIDGE:
#         prior_stds[i] = 0.001# 0.001 to krataw
#     elif dev == ElectricalAppliances.KETTLE:
#         prior_stds[i] = 0.1# to krataw
#     elif dev == ElectricalAppliances.MICROWAVE:
#         prior_stds[i] = 0.001#0.1# 0.001 sto 15 kai paei kala

# prior_noise_std = 1 - sum(prior_stds)

print('prior_stds: ', prior_stds)
print('prior_means: ', prior_means)
print('prior_distributions: ', prior_distributions)
print('prior_noise: ', prior_noise_std)

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]
#self.alpha*reco_loss + self.beta*info_loss + self.gamma*class_loss
model_hparams = [
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
    {
        'model_name': 'MultiRegressorConvEncoder',
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'complexity_cost_weight': 1e-6, 'bayesian_encoder': False, 'bayesian_regressor': False,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    },
    },
    {
        'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
        # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
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
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
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

experiment = NILMSuperExperiments(project_name='SINGLE_LABEL_BENCHMARK', clean_project=True,
                                  devices=devices, save_timeseries_results=False,
                                  experiment_categories=experiment_categories,
                                  experiment_volume=SupportedExperimentVolumes.COMMON_VOLUME,
                                  experiment_parameters=experiment_parameters,
                                  save_model=False, export_plots=False,
                                  )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
#