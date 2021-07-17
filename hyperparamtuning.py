import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from callbacks.callbacks_factories import TrainerCallbacksFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset
from modules.MyDataSet import MyChunkList
from modules.helpers import create_tree_dir, train_eval

with torch.no_grad():
    torch.cuda.empty_cache()

clean = False
ROOT = 'output'
data_dir = '../Datasets'
train_file_dir = 'benchmark/train/'
test_file_dir = 'benchmark/test/'

dev_list = ['fridge',
            #             'kettle',
            #             'washing machine',
            #             'washer dryer',
            #             'tumble dryer',
            #             'dish washer',
            #             'microwave',
            #             'television',
            #             'computer',
            #             'electric space heater'
            ]
mod_list = [
    #             'SF2P',
    # 'S2P',
    #             'SimpleGru',
    #             'FFED',
    #             'SAED',
    # 'FNET',
    'ShortFNET',
    # 'WGRU',
    # 'ConvFourier',
    # 'VIB_SAED',
    # 'VIBFNET',
    # 'VIBSeq2Point',
    # 'ShortNeuralFourier',
    # 'VIBShortNeuralFourier'
]
# REFIT,5,2014-09-01,2014-10-01
# REFIT,6,2014-09-01,2014-10-01
# REFIT,20,2015-01-01,2015-02-01
cat_list = [x for x in ['Single', 'Multi']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean)

exp_type = 'Single'  # 'Multi'

EPOCHS = 100
ITERATIONS = 1

SAMPLE_PERIOD = 6
WINDOW = 50 * 10
device = 'fridge'
BATCH = 256
windows = [50, 50 * 2, 50 * 4, 50 * 10]

model_hparams = {
    'FNET'     : [
        {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
         'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0.25},
        # {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 64,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 8, 'kernel_size': 5, 'cnn_dim': 128,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 8, 'kernel_size': 5, 'cnn_dim': 64,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 16, 'kernel_size': 5, 'cnn_dim': 64,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
    ],
    'ShortFNET': [
        {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
         'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
        # {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 64,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 8, 'kernel_size': 5, 'cnn_dim': 128,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        # {'depth'    : 8, 'kernel_size': 5, 'cnn_dim': 64,
        #  'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0.25},
        {'depth'    : 16, 'kernel_size': 5, 'cnn_dim': 64,
         'input_dim': WINDOW, 'hidden_dim': 256, 'dropout': 0},
    ]
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
# tests_params

train_file = open('{}base{}TrainSetsInfo_{}'.format(train_file_dir, exp_type, device), 'r')
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

train_size = int(0.8 * len(train_dataset_all))
val_size = len(train_dataset_all) - train_size
train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH,
                          shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH,
                        shuffle=True, num_workers=8)

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
for model_name in mod_list:
    for hparams in model_hparams[model_name]:
        print('#' * 40)
        print('MODEL: ', model_name)
        print('#' * 40)
        for iteration in range(1, ITERATIONS + 1):
            print('#' * 20)
            print('Iteration: ', iteration)
            print('#' * 20)
            experiment_name = '_'.join([device, exp_type, 'Train', train_set, '', ])
            experiments.append(experiment_name)
            train_eval(model_name,
                       train_loader,
                       exp_type,
                       tests_params,
                       SAMPLE_PERIOD,
                       BATCH,
                       experiment_name,
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
                       model_hparams=hparams,
                       val_loader=val_loader,
                       callbacks=[TrainerCallbacksFactory.create_earlystopping()])
