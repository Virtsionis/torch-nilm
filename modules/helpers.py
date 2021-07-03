import os, shutil
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modules.MyTrainer import NILMTrainer
from datasources.datasource import DatasourceFactory
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasources.torchdataset import ElectricityIterableDataset, ElectricityDataset

from pytorch_lightning.loggers import WandbLogger

def create_tree_dir(tree_levels={}, clean=False):
    tree_gen = (level for level in tree_levels)
    level = next(tree_gen)
    end = False
    if level=='root':
        print(level)
        root_path = os.getcwd() + '/' + tree_levels[level]
        if clean and os.path.exists(root_path):
            shutil.rmtree(root_path)
            print('all clean')
        if not os.path.exists(root_path):
            os.mkdir(root_path)

    print(root_path)
    base_paths = [root_path]
    while not end:
        try:
            level = next(tree_gen)
            folders = tree_levels[level]
            if isinstance(folders, list):
                paths = []
                for folder in folders:
                    for base_path in base_paths:
                        path = base_path + '/' + folder
                        if not os.path.exists(path):
                            os.mkdir(path)
                        paths.append(path)
            base_paths = paths
        except:
            end=True
    print(1)

def save_report(root_dir=None, model_name=None, device=None, exp_type=None,
                experiment_name=None, iteration=None, results={},
                preds=None, ground=None):

    root_dir = os.getcwd() + '/' + root_dir
    path = '/'.join([root_dir,'results',device,model_name,
                     exp_type,experiment_name,''])
    report_filename = 'REPORT_' + experiment_name + '.csv'
    data_filename = experiment_name + '_iter_' + str(iteration)+'.csv'

    print('Report saved at: ', path)

    if not os.path.exists(path):
        os.mkdir(path)

    if report_filename in os.listdir(path):
        report = pd.read_csv(path+report_filename)
    else:
        cols = ['recall','f1','precision',
                'accuracy','MAE','RETE']
        report = pd.DataFrame(columns=cols)
    report = report.append(results, ignore_index=True)
    report.fillna(np.nan, inplace=True)
    report.to_csv(path+report_filename, index=False)

    cols = ['ground','preds']
    res_data = pd.DataFrame(list(zip(ground, preds)),
                            columns=cols)
    res_data.to_csv(path+data_filename, index=False)

def display_res(root_dir=None, model_name=None, device=None,
                exp_type=None, experiment_name=None, iteration=None,
                low_lim=None, upper_lim=None):

    if low_lim>upper_lim:
        low_lim, upper_lim = upper_lim, low_lim

    root_dir = os.getcwd() + '/' + root_dir

    path = '/'.join([root_dir,'results',device,model_name,
                     exp_type,experiment_name,''])

    if os.path.exists(path):
        report_filename = 'REPORT_' + experiment_name + '.csv'
        data_filename = experiment_name + '_iter_' + str(iteration)+'.csv'

        report = pd.read_csv(path + report_filename)

        if int(iteration) > 0:
            print(report.iloc[int(iteration)-1:int(iteration)])
        else:
            print(report.iloc[int(iteration)])
        data = pd.read_csv(path + data_filename)
        data['ground'][low_lim:upper_lim].plot.line()
        data['preds'][low_lim:upper_lim].plot.line()


def final_device_report(root_dir=None, model_name=None, device=None, exp_type=None,
                experiment_name=None, iteration=None,):
    pass

def train_model(model_name, train_loader, test_loader,
                epochs=5, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
    """
    trainer = pl.Trainer(gpus=1,max_epochs=epochs)

    model = NILMTrainer(model_name=model_name,**kwargs)
    trainer.fit(model, train_loader)

    test_result = trainer.test(model, test_dataloaders=test_loader)
    metrics = test_result[0]['metrics']
    preds = test_result[0]['preds']

    return model, metrics, preds


def train_eval(model_name, train_loader, exp_type, tests_params,
               sample_period, batch_size,experiment_name, iteration,
               device, mmax, window_size, root_dir, val_loader=None,
               epochs=5, saveReport=True, logger=True, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run.
            It's used to look up the class in "model_dict"
    """
    early_stop_callback = EarlyStopping(
                            monitor='val_loss',
                            patience=epochs,
                            mode='min',
                            min_delta=5e-5,
                            # mode='auto',
                            check_finite=True,
                            # stopping_threshold=0.0002,
                            # divergence_threshold=0.0001,
                            check_on_train_epoch_end=True,
                            verbose=True
                        )

    if logger:
        wandb_logger = WandbLogger(name=experiment_name,project='50 epochs comparison')
        trainer = pl.Trainer(gpus=1,
                             max_epochs=epochs,
                             logger=wandb_logger,
                             callbacks=[early_stop_callback],
                            )
    else:
        trainer = pl.Trainer(gpus=1,
                             max_epochs=epochs,
                             callbacks=[early_stop_callback],
                            )

    model = NILMTrainer(model_name=model_name,**kwargs)

    trainer.fit(model, train_loader, val_loader)

    for i in range(len(tests_params)):

        building = tests_params['test_house'][i]
        dataset = tests_params['test_set'][i]
        dates = tests_params['test_date'][i]
        print(80*'#')
        print('Evaluate house {} of {} for {}'.format(building, dataset, dates))
        print(80*'#')

        datasource = DatasourceFactory.create_datasource(dataset)
        test_dataset = ElectricityDataset(datasource=datasource, building=int(building),
                                          window_size=window_size, device=device, mmax=mmax,
                                          dates=dates, sample_period=sample_period)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8)

        ground = test_dataset.meterchunk
        model.set_ground(ground)

        trainer.test(model, test_dataloaders=test_loader)
        if saveReport:
            test_result = model.get_res()
            results = test_result['metrics']
            preds = test_result['preds']
            final_experiment_name = experiment_name + 'test_' + building + '_' + dataset
            save_report(root_dir, model_name, device, exp_type,final_experiment_name,
                        iteration, results, preds, ground)
            del test_dataset, test_loader, ground, final_experiment_name