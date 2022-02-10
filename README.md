# Torch-NILM: An effective deep learning toolkit for Non Intrusive Load Monitoring in Pytorch

## Description
**Torch-NILM** is the first NILM-specific deep learning toolkit build on **Pytorch** and 
**Python**. The purpose of the toolkit is to help researchers design and execute 
huge amount of experiments and comparisons with ease. It contains **3 APIs** that cover
various needs that are encountered when creating a new architecture such as benchmarking, 
hyperparameter tuning and cross validation. Furthermore, **Torch-NILM** comes with a handy
_reporting_ module that exports a results report in _xlsx_ format alongside with comparative
base plots to visualize the results.

The build-in benchmark methodology contains a series of scenarios with escalating difficulty 
to stress test the generalization capabilities of the tested models. In addition, 
**Torch-NILM**  provides a set of powerful baseline models to conduct comparisons.

The toolkit is compatible with **NILMTK**.

## Experiment setup

Defining an experiment requires only a few lines of code. A template of setting an experiment
is provided in _set_experiment.py_.  

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
The **NILMTK** toolkit is used for reading the data.
All the datasets that are compatible with **NILMT** are supported, but the benchmark
is constructed on end-uses from **REDD**, **REFIT** and **UK DALE**.
It should be noted that the data have to be downloaded manually.
In order to load the data, the files _path_manager.py_ and _datasource.py_ insid _datasources/_ directory should be modified accordingly.

## Dependencies

The code has been developed using python3.8 and the dependencies can be found in 
[requirements.txt](requirements.txt):

- numpy~=1.19.5
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
