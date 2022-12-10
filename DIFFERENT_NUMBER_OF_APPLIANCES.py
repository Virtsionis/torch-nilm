from lab.nilm_experiments import *
from constants.constants import *
from constants.enumerates import *

devs_taus = {
    ElectricalAppliances.KETTLE: 0.7, #0.5
    ElectricalAppliances.MICROWAVE: 0.975,
    ElectricalAppliances.FRIDGE: 0.9,
    ElectricalAppliances.WASHING_MACHINE: 0.5, #0.025,
    ElectricalAppliances.DISH_WASHER: 0.5, #0.1,

    ElectricalAppliances.OVEN: 0.5,
    ElectricalAppliances.COMPUTER: 0.5,
    ElectricalAppliances.TELEVISION: 0.5,
    ElectricalAppliances.IMMERSION_HEATER: 0.5,
    ElectricalAppliances.WATER_PUMP: 0.5,

    ElectricalAppliances.LIGHT: 0.5,
    ElectricalAppliances.TOASTER: 0.5,
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
        prior_stds[i] = 0.15
    if dev == ElectricalAppliances.WASHING_MACHINE:
        prior_stds[i] = 0.15
    elif dev == ElectricalAppliances.FRIDGE:
        prior_stds[i] = 0.001
    elif dev == ElectricalAppliances.KETTLE:
        prior_stds[i] = 0.1
    elif dev == ElectricalAppliances.MICROWAVE:
        prior_stds[i] = 0.01
    else:
        prior_stds[i] = 0.1

prior_noise_std = 1 - sum(prior_stds)

devs = list(devs_taus.keys())
for i in [8]:
    experiment_parameters = {
        EPOCHS: 1,
        ITERATIONS: 5,
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
    devs_scenario = devs[:i]
    print('Scenario: ', devs_scenario)
    taus = [devs_taus[dev] for dev in devs_scenario]
    model_hparams = [
        {
            'model_name': 'UNetNiLM',
            'hparams': {'window_size': None, 'taus': taus, 'num_layers': 5, 'features_start': 8,
                        'n_channels': 1, 'num_classes': len(taus), 'pooling_size': 16,
                        'num_quantiles': 1, 'dropout': 0.1, 'd_model': 128, 'lr': 0.001,
                        'adam_betas': (0.9, 0.98),
                        }
        },
        # {
        #     'model_name': 'CNN1DUnetNilm',
        #     'hparams': {'window_size': None, 'taus': taus, 'num_classes': len(taus),
        #                 'pooling_size': 16, 'num_quantiles': 1, 'dropout': 0.0, 'lr': 0.001,
        #                 'adam_betas': (0.9, 0.98),
        #                 }
        # },

        {
            'model_name': 'VariationalMultiRegressorConvEncoder',
            'hparams': {'input_dim': None, 'distribution_dim': 16, 'targets_num': len(devs_scenario),
                        'beta': 1e-2, 'gamma': 1e-0, 'complexity_cost_weight': 1e-6,
                        'dae_output_dim': experiment_parameters[FIXED_WINDOW],
                        'max_noise': 0.1, 'prior_stds': prior_stds, 'prior_noise_std': prior_noise_std,
                        'prior_means': prior_means, 'prior_distributions': prior_distributions,
                        'lr': 1e-3, 'bayesian_encoder': False, 'bayesian_regressor': False,
                        },
        },
    ]

    model_hparams = ModelHyperModelParameters(model_hparams)
    experiment_parameters = ExperimentParameters(**experiment_parameters)

    experiment = NILMSuperExperiments(project_name='MULTI_TARGET_{}_Appliances'.format(str(i)), clean_project=True,
                                      devices=devs_scenario, save_timeseries_results=False,
                                      experiment_categories=experiment_categories,
                                      experiment_volume=SupportedExperimentVolumes.UKDALE_ALL_VOLUME,
                                      experiment_parameters=experiment_parameters,
                                      save_model=False, export_plots=False,
                                      )

    experiment.run_benchmark(model_hparams=model_hparams)
    print('#' * 80)
# experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)
#
