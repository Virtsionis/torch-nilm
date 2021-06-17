import torch
import pandas as pd

from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset
from modules.MyTrainer import NILMTrainer
from modules.helpers import create_tree_dir, save_report, train_model, display_res, train_eval
from torch.utils.data import Dataset, DataLoader

from modules.MyDataSet import MyChunk, MyChunkList

with torch.no_grad():
    torch.cuda.empty_cache()

clean = True
ROOT = 'output'
data_dir = '../Datasets'
train_file_dir = 'dates/train/'
test_file_dir = 'dates/test/'

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
    #             'S2P',
    #             'SimpleGru',
    #             'FFED',
    #             'SAED',
    #             'FNET',
    #             'WGRU',
    'ConvFourier'
]
cat_list = [x for x in ['Single', 'Multi']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean)

exp_type = 'Single'  # 'Multi'

EPOCHS = 5
ITERATIONS = 1

SAMPLE_PERIOD = 6
WINDOW = 50
device = 'fridge'
BATCH = 1024

model_hparams = {
    'SimpleGru'  : {},
    'SAED'       : {'window_size': WINDOW},
    'FFED'       : {},
    'WGRU'       : {'dropout': 0.25},
    'S2P'        : {'window_size': WINDOW, 'dropout': 0.25},
    'ConvFourier': {'window_size': WINDOW, 'dropout': 0.25},
    'SF2P'       : {'window_size': WINDOW, 'dropout': 0.25},
    'FNET'       : {'depth'    : 2, 'kernel_size': 5, 'cnn_dim': 64,
                    'input_dim': WINDOW, 'hidden_dim': WINDOW * 8, 'dropout': 0.25}
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
    train_dataset = ElectricityDataset(datasource=datasource,
                                       building=int(train_house),
                                       device=device,
                                       start_date=train_dates[0],
                                       end_date=train_dates[1],
                                       transform=None,
                                       window_size=WINDOW,
                                       mmax=None,
                                       sample_period=SAMPLE_PERIOD,
                                       chunksize=1000*BATCH,
                                       batch_size=BATCH)
    # train_dataset = MyChunk(path=path, building=int(train_house), window_size=WINDOW,
    #                         device=device, dates=train_dates, sample_period=SAMPLE_PERIOD)
else:
    for line in train_file:
        toks = line.split(',')
        train_set = toks[0]
        break
    train_file.close()
    train_dataset = MyChunkList(device, filename=train_file,
                                window_size=WINDOW, sample_period=SAMPLE_PERIOD)

train_loader = DataLoader(train_dataset, batch_size=BATCH,
                          shuffle=False, num_workers=1)
mmax = train_dataset.mmax

experiments = []
experiment_name = '_'.join([device, exp_type, 'Train', train_set, '', ])
print(experiment_name)
eval_params = {'device'     : device,
               'mmax'       : mmax,
               'groundtruth': ''}
for model_name in mod_list:
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
                   WINDOW,
                   ROOT,
                   data_dir,
                   epochs=EPOCHS,
                   eval_params=eval_params,
                   model_hparams=model_hparams[model_name])
