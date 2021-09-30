import torch
import pandas as pd
from callbacks.callbacks_factories import TrainerCallbacksFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityMultiBuildingsDataset
from modules.helpers import create_tree_dir, train_eval, create_time_folds
from torch.utils.data import DataLoader, random_split

with torch.no_grad():
    torch.cuda.empty_cache()

clean = True
PLOTS = False
ROOT = 'Single_Building_3fold_CV'
exp_volume = 'cv'
# data_dir = '/mnt/B40864F10864B450/WorkSpace/PHD/PHD_exps/data'
data_dir = '../Datasets'

train_file_dir = 'benchmark/{}/train/'.format(exp_volume)

dev_list = [
                        'dish washer',
                        # 'washing machine',
                        # 'kettle',
                        # 'fridge',
                        # 'microwave',
                        # 'computer',
                        # 'television',
                        # 'tumble dryer',
                        # 'electric space heater',
            ]
mod_list = [
    # 'VIB_SAED',
    # 'VIBWGRU',
    # 'VIBFNET',
    # 'VIBShortFNET',
    # 'VIBSeq2Point',
    # 'FNET',
    # 'S2P',
    # 'SimpleGru',
    # 'SAED',
    # 'WGRU',
    # 'VAE',
    'DAE',
]

cat_list = [x for x in ['Single', 'Multi']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean, plots=PLOTS)

exp_type = 'Single'

EPOCHS = 1
CV_FOLDS = 3
SAMPLE_PERIOD = 6
WINDOW = 256
BATCH = 256

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

    'VIBSeq2Point'         : {'window_size': WINDOW, 'dropout': 0},
    'VIB_SAED'             : {'window_size': WINDOW},
    'VIBWGRU'              : {'dropout': 0.25},
    'VIBFNET'              : {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
                              'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
    'VIBShortFNET'         : {'depth'    : 1, 'kernel_size': 5, 'cnn_dim': 128,
                              'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},

    'VAE'             : {'sequence_len': WINDOW, 'dropout':0.2},
    'DAE'             : {'sequence_len': WINDOW, 'dropout':0.2},
    }

    for model_name in mod_list:
        print('#' * 40)
        print('MODEL: ', model_name)
        print('#' * 40)

        if model_name in ['VAE', 'DAE']:
            rolling_window = False
            WINDOW -= WINDOW % 8
            print('NEW WINDOW =', WINDOW)
            if WINDOW>BATCH:
                BATCH=WINDOW
        else:
            rolling_window = True

        file = open('{}base{}TrainSetsInfo_{}'.format(train_file_dir, exp_type, device), 'r')
        for line in file:
            toks = line.split(',')
            train_set = toks[0]
            train_house = toks[1]
            dates = [str(toks[2]), str(toks[3].rstrip("\n"))]
            break
        file.close()

        datasource = DatasourceFactory.create_datasource(train_set)
        time_folds = create_time_folds(start_date=dates[0], end_date=dates[1],\
                                       folds=CV_FOLDS, drop_last=False)

        for fold in range(CV_FOLDS):
            print('#'*80)
            print('TRAIN FOR FOLD {}'.format(fold))
            print('#'*80)
            data = {'test_house': [train_house], 'test_set': [train_set],\
                    'test_date': [time_folds[fold]['test_dates']]}
            print(data)

            tests_params = pd.DataFrame(data)
            print(tests_params)
            train_dates = time_folds[fold]['train_dates']
            train_info = []
            for train_date in train_dates:
                if len(train_date):
                    train_info.append({
                                    'device' : device,
                                    'datasource' : datasource,
                                    'building' : int(train_house),
                                    'dates' : train_date,
                                    })
            print(train_info)
            train_dataset_all = ElectricityMultiBuildingsDataset(train_info,
                                                                device=device,
                                                                window_size=WINDOW,
                                                                rolling_window=rolling_window,
                                                                sample_period=SAMPLE_PERIOD)

            train_size = int(0.8 * len(train_dataset_all))
            val_size = len(train_dataset_all) - train_size
            train_dataset, val_dataset = random_split(train_dataset_all,
                                                      [train_size, val_size],
                                                      generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_dataset, batch_size=BATCH,
                                        shuffle=True, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=BATCH,
                                shuffle=True, num_workers=8)
            mmax = train_dataset_all.mmax
            means = train_dataset_all.means
            stds = train_dataset_all.stds

            experiment_name = '_'.join([device, exp_type, 'Train', train_set, '', ])
            print(experiment_name)
            eval_params = {
                'device'     : device,
                'mmax'       : mmax,
                'means'      : train_dataset_all.meter_means,
                'stds'       : train_dataset_all.meter_stds,
                'groundtruth': ''
            }
            train_eval(model_name,
                       train_loader,
                       exp_type,
                       tests_params,
                       SAMPLE_PERIOD,
                       BATCH,
                       experiment_name,
                       exp_volume,
                       fold,
                       device,
                       mmax,
                       means,
                       stds,
                       train_dataset_all.meter_means,
                       train_dataset_all.meter_stds,
                       WINDOW,
                       ROOT,
                       data_dir,
                       rolling_window=rolling_window,
                       epochs=EPOCHS,
                       eval_params=eval_params,
                       val_loader=val_loader,
                       model_hparams=model_hparams[model_name],
                       plots=PLOTS,
                       callbacks=[TrainerCallbacksFactory.create_earlystopping()]
                       )
# get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='FINAL_REPORT_{}'.format(ROOT))

