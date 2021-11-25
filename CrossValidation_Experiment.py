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
                        # 'dish washer',
                        # 'washing machine',
                        'kettle',
                        # 'fridge',
                        # 'microwave',
                        # 'computer',
                        # 'television',
                        # 'tumble dryer',
                        # 'electric space heater',
            ]
mod_list = [
    'FNET',
    # 'S2P',
    # 'SAED',
    # 'WGRU',
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
        'SAED'                 : {'window_size': WINDOW},
        'WGRU'                 : {'dropout': 0.25},
        'S2P'                  : {'window_size': WINDOW, 'dropout': 0.25},
        'FNET'                 : {'depth'    : 4, 'kernel_size': 5, 'cnn_dim': 128,
                                  'input_dim': WINDOW, 'hidden_dim': WINDOW * 4, 'dropout': 0},
    }

    for model_name in mod_list:
        print('#' * 40)
        print('MODEL: ', model_name)
        print('#' * 40)

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
            train_dataset_all = ElectricityMultiBuildingsDataset(train_info=train_info,
                                                                 window_size=WINDOW,
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
