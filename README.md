# Torch-NILM: An effective deep learning toolkit for Non Intrusive Load Monitoring in Pytorch

## Description
**[Torch-NILM](https://doi.org/10.3390/en15072647)** is the first NILM-specific deep learning toolkit build on **[Pytorch](https://pytorch.org)** and 
**[Python](https://www.python.org)**. The purpose of the toolkit is to help researchers design and execute 
huge amount of experiments and comparisons with ease. It contains **3 APIs** that cover
various needs that are encountered when creating a new architecture such as benchmarking, 
hyperparameter tuning and cross validation. Furthermore, **Torch-NILM** comes with a handy
_reporting_ module that exports a results report in _xlsx_ format alongside with comparative
base plots to visualize the results.

The build-in benchmark methodology contains a series of scenarios with escalating difficulty 
to stress test the generalization capabilities of the tested models. In addition, 
**[Torch-NILM](https://doi.org/10.3390/en15072647)**  provides a set of powerful baseline models to conduct comparisons.

The toolkit is compatible with **[NILMTK](https://nilmtk.github.io)**.

The complementary paper "Torch-NILM: An Effective Deep Learning Toolkit for Non-Intrusive Load Monitoring in Pytorch" can be found [here](https://www.mdpi.com/1996-1073/15/7/2647).

## Experiment guide

Defining an experiment requires only a few lines of code. A template of setting an experiment
is provided in _set_experiment.py_.  

### Configurations setup
In order to set up the experiment the following configurations should be provided:

a) The _experiment_parameters_ contain the general experiment configurations that are essential
for all the supported experiments. A small description is provided bellow. For more information go
to the corresponding doc string in _torch_nilm_/_lab_/_nilm_experiments.py_ or consult the paper.

    - EPOCHS: 'The number of training epochs of a model'
    - ITERATIONS: 'The number of iterations each experiment should run. It is helpful for calculating 
    statistics'
    - INFERENCE_CPU: 'If _true_ the inference will executed on the CPU'
    - SAMPLE_PERIOD: 'The sampling period in seconds'
    - BATCH_SIZE: 'The batch size needed for training and inference of the neural networks'
    - ITERABLE_DATASET: 'If _True_ the data will be provided to the network in an efficient way. 
    More in https://pytorch.org/docs/stable/data.html'
    - PREPROCESSING_METHOD: 'The preprocessing method. Four methods are supported: sequence-to-sequence
    learning, sliding-window approach, midpoint-window method and sequence-to-subsequence approach.'
    - FILLNA_METHOD: 'The method to fill missing values. Zero filling and linear interpolation are supported'
    - FIXED_WINDOW: 'The length of the input sequence' 
    - SUBSEQ_WINDOW: 'The length of the output sequence when sequence-to-subsequence preprocessing method is chosen.'
    - TRAIN_TEST_SPLIT: 'The ratio of data to used for training/inference'
    - CV_FOLDS: 'The number of folds when cross validation experiment is chosen'
    - NOISE_FACTOR: 'The percentage of the added noise can controlled with a noise factor, a factor to multiply a 
    gaussian noise signal, which will be added to the normalized mains timeseries.'

After the declaration of the _experiment_parameters_ list the user should save the list as an _ExperimentParameters_ 
object:
    
    experiment_parameters = ExperimentParameters(**experiment_parameters)

b) The _devices_ list contains the desired electrical devices/appliances to run the experiments for. Currently, five
appliances are supported:

    ElectricalAppliances.KETTLE
    ElectricalAppliances.MICROWAVE
    ElectricalAppliances.FRIDGE
    ElectricalAppliances.WASHING_MACHINE
    ElectricalAppliances.DISH_WASHER


c) The _experiment_categories_ list contains the desired benchmark categories to be executed. The categories are 
based on the benchmark method proposed in [1]. Two categories are supported:

    SupportedExperimentCategories.SINGLE_CATEGORY
    SupportedExperimentCategories.MULTI_CATEGORY

d) The _model_hparams_ list contains the hyperparameters for the desired models to train.
The user can add only the desired models. 

    model_hparams = [
        {
            'model_name': 'VAE',
            'hparams': {'window_size': None, 'cnn_dim': 256, 'kernel_size': 3, 'latent_dim': 16},
        },
        {
                    'model_name': 'NFED',
                    'hparams': {'depth': 1, 'kernel_size': 5, 'cnn_dim': 128,
                                'input_dim': None, 'hidden_dim': 256, 'dropout': 0.0},
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
    ]

After the declaration of the _model_hparams_ list the user should save the list as an _ModelHyperModelParameters_ object:
    
    model_hparams = ModelHyperModelParameters(model_hparams)

e) In order to execute hyperparameter tuning with cross validation, the user should define the _hparam_tuning_ list. That
list contains the versions of the desired neural under test.

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

After the declaration of the _hparam_tuning_ list the user should save the list as an _HyperParameterTuning_ object:
   
    hparam_tuning = HyperParameterTuning(hparam_tuning)

### Experiments setup

a) In order to run the experiments a _NILMExperiments_ object should be defined as shown bellow. 

    experiment = NILMExperiments(
            project_name='MyProject', // the project name 
            clean_project=True, // whether to delete the folders under the project name or not
            devices=devices, // the device list
            save_timeseries_results=False, // whether to save the network's output or not
            experiment_categories=experiment_categories, // the experiment categories
            experiment_volume=SupportedExperimentVolumes.LARGE_VOLUME,// the data volume
            experiment_parameters=experiment_parameters, // the general experiment parameters
            save_model=True, // whether to save model weights or not  
            export_plots=True,// whether to export result plots or not
    )

b) This experiments object contains all the experiment APIs, which can be called as shown bellow.

    experiment.run_benchmark(model_hparams=model_hparams)
    experiment.run_cross_validation(model_hparams=model_hparams)
    experiment.run_hyperparameter_tuning_cross_validation(hparam_tuning=hparam_tuning)

c) After an experiment is executed the corresponding _statistical_report.xlsx_ and the plots are exported in the 
_project_name_/_results_/ directory.

d) If for some reason an experiment was interrupted and the report was not created the user can run the _export_report_ 
API with the same settings as the desired experiment API. This way a new report will be created without re-running the
experiment.

    ## experiment.run_benchmark(model_hparams=model_hparams)
    experiment.export_report(model_hparams=model_hparams, experiment_type=SupportedNilmExperiments.BENCHMARK)

## Run Experiment

In order to execute the experiment run: 
```python
python3 set_experiment.py
```
The results of each API are saved under the directory _output_/_project-name_/_results_/_api-name_. 
Inside that directory 3 folders exist (depending the user configurations); plots, results and 
saved_models. Plots directory contains all the produced graphs. Results contains the results folders
for every appliance and model alongside the final statistical report in _xlsx_ format. Saved_models
contains all the weights of the models for every appliance and iteration.

## Datasets
The **NILMTK**[2] toolkit is used for reading the data.
All the datasets that are compatible with **NILMTK** are supported, but the benchmark
is constructed on end-uses from **UK DALE**[3], **REDD**[4] and **REFIT**[5]. 
It should be noted that the data have to be downloaded manually.
In order to load the data, the files _path_manager.py_ and _datasource.py_ inside _datasources/_ directory should be 
modified accordingly.

## Dependencies

The code has been developed using python3.8 and the dependencies can be found in 
[requirements.txt](requirements.txt):

- numpy>=1.21
- pandas~=0.25.3
- torch~=1.9.0+cu111
- plotly~=5.4.0
- setuptools~=49.6.0
- wandb~=0.10.32
- fuzzywuzzy~=0.18.0
- nilmtk~=0.4.3
- loguru~=0.5.3
- numba~=0.53.1
- scikit-image~=0.18.2
- h5py~=3.4.0
- dash~=2.0.0

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## References
1. Symeonidis, N.; Nalmpantis, C.; Vrakas, D. A Benchmark Framework to Evaluate Energy Disaggregation Solutions. International 541
Conference on Engineering Applications of Neural Networks. Springer, 2019, pp. 19–30.
2. Batra, N.; Kelly, J.; Parson, O.; Dutta, H.; Knottenbelt, W.; Rogers, A.; Singh, A.; Srivastava, M. NILMTK: an open source toolkit 525
for non-intrusive load monitoring. Proceedings of the 5th international conference on Future energy systems, 2014, pp. 265–276.
3. Jack, K.; William, K. The UK-DALE dataset domestic appliance-level electricity demand and whole-house demand from five UK
homes. Sci. Data 2015, 2, 150007.
4. Kolter, J.Z.; Johnson, M.J. REDD: A public data set for energy disaggregation research. Workshop on data mining applications in
sustainability (SIGKDD), San Diego, CA, 2011, Vol. 25, pp. 59–62.
5. Firth, S.; Kane, T.; Dimitriou, V.; Hassan, T.; Fouchal, F.; Coleman, M.; Webb, L. REFIT Smart Home dataset, 2017.
doi:10.17028/rd.lboro.2070091.v1.
