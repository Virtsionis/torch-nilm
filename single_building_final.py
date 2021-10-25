import torch
import pandas as pd

from callbacks.callbacks_factories import TrainerCallbacksFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset, ElectricityMultiBuildingsDataset
from modules.helpers import create_tree_dir, train_eval, get_final_report
from torch.utils.data import DataLoader, random_split


with torch.no_grad():
    torch.cuda.empty_cache()

clean = False
PLOTS = False
SAVE_TIMESERIES = False
ROOT = 'FNET_VIB_SINGLE_BUILDINGS'
exp_volume = 'large'
data_dir = '../Datasets'
train_file_dir = 'benchmark/{}/train/'.format(exp_volume)
test_file_dir = 'benchmark/{}/test/'.format(exp_volume)

dev_list = [
                        # 'microwave',
                        # 'kettle',
                        # 'fridge',
                        'washing machine',
                        # 'dish washer',
            ]

mod_list = [
    'VIBFNET',
    #  'FNET',
    #  'S2P',
    #  'SAED',
    #  'WGRU',
    #  'SimpleGru',
]

WINDOWS = {
    'VIBFNET': {
        'microwave'            : 50,
        'kettle'               : 50,
        'fridge'               : 350,
        'washing machine'      : 150,
        'dish washer'          : 450,
    },
    'FNET': {
        'microwave'            : 50,
        'kettle'               : 50,
        'fridge'               : 350,
        'washing machine'      : 150,
        'dish washer'          : 450,
    },
    'S2P': {
        'microwave'            : 100,
        'kettle'               : 300,
        'fridge'               : 400,
        'washing machine'      : 400,
        'dish washer'          : 500,
    },
    'WGRU': {
        'microwave'            : 100,
        'kettle'               : 150,
        'fridge'               : 450,
        'washing machine'      : 150,
        'dish washer'          : 350,
    },
    'SAED': {
        'microwave'            : 100,
        'kettle'               : 50,
        'fridge'               : 250,
        'washing machine'      : 50,
        'dish washer'          : 200,
    },
    'SimpleGru': {
        'microwave'            : 100,
        'kettle'               : 200,
        'fridge'               : 400,
        'washing machine'      : 200,
        'dish washer'          : 450,
    },
}

cat_list = [x for x in ['Single']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean, plots=PLOTS)

exp_type = 'Single'

EPOCHS = 100
ITERATIONS = 5
SAMPLE_PERIOD = 6
BATCH = 512

for model_name in mod_list:
    windows = WINDOWS[model_name]
    for device in dev_list:
        WINDOW = windows[device]
        print('#' * 160)
        print('MODEL: {}, DEVICE: {}, Window: {}'.format(model_name, device, WINDOW))
        print('#' * 160)
        model_hparams = {
            'SimpleGru'            : {},
            'SAED'                 : {'window_size': WINDOW},
            'WGRU'                 : {'dropout': 0},
            'S2P'                  : {'window_size': WINDOW, 'dropout': 0},
            'FNET'                 : {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
                                    'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0},
            'VIBFNET'              : {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128, 'max_noise':.9,
                                     'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.0, 'beta':1e-5},
        }

        test_houses = []
        test_sets = []
        test_dates = []
        test_file = open('{}base{}TestSetsInfo_{}'.format(test_file_dir, exp_type, device), 'r')
        for line in test_file:
            toks = line.split(',')
            test_sets.append(toks[0])
            test_houses.append(toks[1])
            test_dates.append([str(toks[2]), str(toks[3].rstrip("\n"))])
        test_file.close()
        data = {'test_house': test_houses, 'test_set': test_sets, 'test_date': test_dates}
        tests_params = pd.DataFrame(data)

        train_file = open('{}base{}TrainSetsInfo_{}'.format(train_file_dir, exp_type, device), 'r')
        if exp_type == 'Single':
            for line in train_file:
                toks = line.split(',')
                train_set = toks[0]
                train_house = toks[1]
                train_dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
                break
            train_file.close()

            path = data_dir + '/{}/{}.h5'.format(train_set, train_set)
            print(path)
            datasource = DatasourceFactory.create_datasource(train_set)
            train_dataset_all = ElectricityDataset(datasource=datasource, building=int(train_house), window_size=WINDOW,
                                                device=device, dates=train_dates, sample_period=SAMPLE_PERIOD)

        else:
            train_info = []
            index = 0
            for line in train_file:
                toks = line.split(',')
                train_set = toks[0]
                train_house = toks[1]
                train_dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
                path = data_dir + '/{}/{}.h5'.format(train_set, train_set)
                datasource = DatasourceFactory.create_datasource(train_set)
                train_info.append({
                                    'device' : device,
                                    'datasource' : datasource,
                                    'building' : int(train_house),
                                    'dates' : train_dates,
                                }
                                )
            train_file.close()
            train_dataset_all = ElectricityMultiBuildingsDataset(train_info,
                                                                device=device,
                                                                window_size=WINDOW,
                                                                sample_period=SAMPLE_PERIOD)
        train_size = int(0.8 * len(train_dataset_all))
        val_size = len(train_dataset_all) - train_size
        train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=BATCH,
                                  shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH,
                                shuffle=False, num_workers=8)
        mmax = train_dataset_all.mmax
        means = train_dataset_all.means
        stds = train_dataset_all.stds

        experiments = []
        experiment_name = '_'.join([device, exp_type, 'Train', train_set, '', ])
        print(experiment_name)
        eval_params = {'device'     : device,
                    'mmax'       : mmax,
                    'means'      : train_dataset_all.meter_means,
                    'stds'       : train_dataset_all.meter_stds,
                    'groundtruth': ''}


        for iteration in range(1, ITERATIONS + 1):
            print('#' * 20)
            print('Iteration: ', iteration)
            print('#' * 20)
            experiment_name = '_'.join([device, exp_type, 'Train', train_set, '', ])
            train_eval(model_name,
                    train_loader,
                    exp_type,
                    tests_params,
                    SAMPLE_PERIOD,
                    BATCH,
                    experiment_name,
                    exp_volume,
                    iteration,
                    device,
                    mmax,
                    means,
                    stds,
                    train_dataset_all.meter_means,
                    train_dataset_all.meter_stds,
                    WINDOW,
                    ROOT,
                    data_dir,
                    epochs=EPOCHS,
                    eval_params=eval_params,
                    model_hparams=model_hparams[model_name],
                    val_loader=val_loader,
                    plots=PLOTS,
                    save_timeseries=SAVE_TIMESERIES,
                    callbacks=[TrainerCallbacksFactory.create_earlystopping()]
                    )
