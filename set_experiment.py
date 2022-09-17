from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *
experiment_parameters = {
    EPOCHS: 100,
    ITERATIONS: 3,
    INFERENCE_CPU: False,
    SAMPLE_PERIOD: 10,
    BATCH_SIZE: 1024,
    ITERABLE_DATASET: False,
    PREPROCESSING_METHOD: SupportedPreprocessingMethods.ROLLING_WINDOW,
    # SCALING_METHOD: SupportedScalingMethods.NORMALIZATION,
    FIXED_WINDOW: 128,
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
]

prior_weights = [0.1 for i in range(0, len(devices))]
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
        prior_weights[i] = 0.01#0.1# 0.001 sto 15 kai paei kala

prior_noise = 1 - sum(prior_weights)

print('prior_weights: ', prior_weights)
print('prior_noise: ', prior_noise)

experiment_categories = [
    SupportedExperimentCategories.SINGLE_CATEGORY,
]

model_hparams = [
    {
        'model_name': 'VAE',
        'hparams': {'window_size': None, 'cnn_dim': 256, 'kernel_size': 3, 'latent_dim': 16},
    },
    {
        'model_name': 'SuperVAE',
        'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
                    'alpha': 1e-2, 'beta': 1e-3, 'gamma': 1e-1, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                    'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,
                    },
    },
]

hparam_tuning = [
    {
        'model_name': 'SuperVAE',
        'hparams': [
            {'input_dim': None, 'distribution_dim': 8, 'targets_num': len(devices),
             'alpha': 1, 'beta': 1e-5, 'gamma': 1e-2, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
             'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise, },
            {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devices),
             'alpha': 1, 'beta': 1e-5, 'gamma': 1e-2, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
             'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise, },
            {'input_dim': None, 'distribution_dim': 32, 'targets_num': len(devices),
             'alpha': 1, 'beta': 1e-5, 'gamma': 1e-2, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
             'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,},
            {'input_dim': None, 'distribution_dim': 64, 'targets_num': len(devices),
             'alpha': 1, 'beta': 1e-5, 'gamma': 1e-2, 'dae_output_dim': experiment_parameters[FIXED_WINDOW],
             'max_noise': 0.1, 'prior_weights': prior_weights, 'prior_noise': prior_noise,},
        ]
    },
]

model_hparams = ModelHyperModelParameters(model_hparams)
hparam_tuning = HyperParameterTuning(hparam_tuning)
experiment_parameters = ExperimentParameters(**experiment_parameters)

# SuperVae21Unet-> to kalytero, only noise info loss

# experiment = NILMSuperExperiments(project_name='IssueGit', clean_project=False,
#                                   devices=devices, save_timeseries_results=False,
#                                   experiment_categories=experiment_categories,
#                                   experiment_volume=SupportedExperimentVolumes.COMMON_VOLUME,
#                                   experiment_parameters=experiment_parameters,
#                                   save_model=False, export_plots=False,
#                                   )

experiment = NILMExperiments(project_name='IssueGit', clean_project=True,
                             devices=devices, save_timeseries_results=False, experiment_categories=experiment_categories,
                             experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,
                             experiment_parameters=experiment_parameters,
                             save_model=False, export_plots=False,
                             )

experiment.run_benchmark(model_hparams=model_hparams)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
# experiment.run_cross_validation(model_hparams=model_hparams)
# experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)
# experiment.export_report(hparam_tuning=hparam_tuning, experiment_type=SupportedNilmExperiments.HYPERPARAM_TUNE_CV)
