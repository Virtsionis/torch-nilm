import torch
import pandas as pd

from callbacks.callbacks_factories import TrainerCallbacksFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset, ElectricityMultiBuildingsDataset
from modules.helpers import create_tree_dir, save_report, train_model,\
        display_res, train_eval, get_final_report
from torch.utils.data import Dataset, DataLoader, random_split

from modules.MyDataSet import MyChunk, MyChunkList

with torch.no_grad():
    torch.cuda.empty_cache()

clean = False
PLOTS = True
ROOT = 'test_multi'#'visuals2'
exp_volume = 'large'
data_dir = '/mnt/B40864F10864B450/WorkSpace/PHD/PHD_exps/data'
train_file_dir = 'benchmark/{}/train/'.format(exp_volume)
test_file_dir = 'benchmark/{}/test/'.format(exp_volume)

dev_list = [
                        # 'electric space heater',
                        'television',
                        'computer',
                        'washing machine',
                        'kettle',
                        'dish washer',
                        'fridge',
                        'microwave',
                        # 'tumble dryer',
            ]
mod_list = [
    #             'SF2P',
    # 'S2P',
                'SimpleGru',
    #             'FFED',
                # 'SAED',
    # 'FNET',
    # 'ShortFNET',
    # 'PosFNET',
    # 'ShortPosFNET',
    # 'WGRU',
    # 'ConvFourier',
    # 'VIB_SAED',
    # 'VIBFNET',
    # 'VIBSeq2Point',
    # 'ShortNeuralFourier',
    # 'VIBShortNeuralFourier',
    # 'BayesSimpleGru',
    # 'BayesFNET',
    # 'BERT4NILM',
    # 'BayesWGRU',
    # 'BayesSeq2Point',
]
# REFIT,5,2014-09-01,2014-10-01
# REFIT,6,2014-09-01,2014-10-01
# REFIT,20,2015-01-01,2015-02-01
cat_list = [x for x in ['Single', 'Multi']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean, plots=PLOTS)

exp_type = 'Multi'  # 'Single'

EPOCHS = 1
ITERATIONS = 1

SAMPLE_PERIOD = 6
WINDOW = 50
# device = 'fridge'
# BATCH = 256
BATCH = 1000
for device in dev_list:
    print('#' * 160)
    print('DEVICE: ', device)
    print('#' * 160)
    model_hparams = {
        'SimpleGru'            : {},
        'SAED'                 : {'window_size': WINDOW},
        'FFED'                 : {},
        'WGRU'                 : {'dropout': 0.25},
        'S2P'                  : {'window_size': WINDOW, 'dropout': 0.25},
        'ConvFourier'          : {'window_size': WINDOW, 'dropout': 0.25},
        'SF2P'                 : {'window_size': WINDOW, 'dropout': 0.25},
        'FNET'                 : {'depth'    : 4, 'kernel_size': 5, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
        'PosFNET'                 : {'depth'    : 8, 'kernel_size': 5, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 10, 'dropout': 0},
        'ShortFNET'            : {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
        'ShortPosFNET'            : {'depth'    : 2, 'kernel_size': 5, 'cnn_dim': 64,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
        'VIBSeq2Point'         : {'window_size': WINDOW, 'dropout': 0},
        'VIB_SAED'             : {'window_size': WINDOW},
        'VIBFNET'              : {'depth'    : 16, 'kernel_size': 2, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 2, 'dropout': 0},
        'ShortNeuralFourier'   : {'window_size': WINDOW},
        'VIBShortNeuralFourier': {'window_size': WINDOW},
        'BERT4NILM': {'window_size':WINDOW,'drop_out':0.5,'output_size':1,
                      'hidden':256,'heads':2,'n_layers':2},
        'BayesSimpleGru': {},
        'BayesWGRU'                 : {'dropout': 0.0},
        'BayesSeq2Point': {'window_size': WINDOW},
        # 'BayesFNET'                 : {'depth'    : 6, 'kernel_size': 5, 'cnn_dim': 128,
        #                           'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},#kettle
        'BayesFNET'                 : {'depth'    : 6, 'kernel_size': 5, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': 500, 'dropout': 0},#fridge

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
        print('#' * 40)
        print('MODEL: ', model_name)
        print('#' * 40)
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
                       callbacks=[TrainerCallbacksFactory.create_earlystopping()]
                       )
get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='FINAL_REPORT')