from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
experiment_parameters = {
    EPOCHS: 100,
    ITERATIONS: 3,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 6,
    BATCH_SIZE: 512,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    # SCALING_METHOD: SupportedScalingMethods.NORMALIZATION,
    FIXED_WINDOW: 100,
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
    # ElectricalAppliances.LIGHT,
]

prior_weights = [0.01 for i in range(0, len(devices))]
prior_noise = 0.01
for i, dev in enumerate(devices):
    if dev == ElectricalAppliances.DISH_WASHER:
        prior_weights[i] = 0.15# 0.1 sto 15 kai paei kala
    elif dev == ElectricalAppliances.WASHING_MACHINE:
        prior_weights[i] = 0.15#0.1 to krataw
    elif dev == ElectricalAppliances.FRIDGE:
        prior_weights[i] = 0.001# 0.001 to krataw
    elif dev == ElectricalAppliances.KETTLE:
        prior_weights[i] = 0.1# to krataw
    elif dev == ElectricalAppliances.MICROWAVE:
        prior_weights[i] = 0.001#0.1# 0.001 sto 15 kai paei kala

prior_noise = 1 - sum(prior_weights)

print('prior_weights: ', prior_weights)
print('prior_noise: ', prior_noise)

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]

model_hparams = [
    # {
    #     'model_name': 'SuperVAE',# FROM LATENT SPACE
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-1, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
    #                 },
    # },
    {
        'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
                                     # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'alpha': 0*1e-2, 'beta': 1e-3, 'gamma': 1e-1, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
                    },
    },
    {
        'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
        # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'alpha': 1e-2, 'beta': 1e-1, 'gamma': 1e-1,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
                    },
    },
    {
        'model_name': 'SuperVAE1b',  # FROM LATENT SPACE but with 2 changes
        # a) deeper shallow nets, b) got rid of reshape layers
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'alpha': 1e-1, 'beta': 1e-1, 'gamma': 1e-1,
                    'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
                    },
    },
    # {
    #     'model_name': 'SuperVAE2',# FROM PRIORS
    #     'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
    #                 'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-1, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
    #                 'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
    #                 },
    # },
]


model_hparams = ModelHyperModelParameters(model_hparams)
experiment_parameters = ExperimentParameters(**experiment_parameters)

experiment = NILMSuperExperiments(project_name='SuperVaeMultiLabelMINE_1b_', clean_project=False,
                                  devices=devices, save_timeseries_results=False,
                                  experiment_categories=experiment_categories,
                                  experiment_volume=SupportedExperimentVolumes.COMMON_VOLUME,
                                  experiment_parameters=experiment_parameters,
                                  save_model=False, export_plots=False,
                                  )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
