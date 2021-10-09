import os, shutil
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lab.training_tools import TrainingToolsFactory
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityIterableDataset, ElectricityDataset

def create_tree_dir(tree_levels={}, clean=False, plots=True):
    tree_gen = (level for level in tree_levels)
    level = next(tree_gen)
    end = False
    if level == 'root':
        # print(level)
        root_path = os.getcwd() + '/' + tree_levels[level]
        if clean and os.path.exists(root_path):
            shutil.rmtree(root_path)
            print('all clean')
        if not os.path.exists(root_path):
            os.mkdir(root_path)

    # print(root_path)
    base_paths = [root_path]
    if plots:
        plot_path = root_path + '/plots'
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

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
            end = True
    print(1)

def save_report(root_dir=None, model_name=None, device=None, exp_type=None, save_timeseries=True,
                experiment_name=None, exp_volume='large', iteration=None, results={},
                preds=None, ground=None, model_hparams=None, epochs=None, plots=True):

    root_dir = os.getcwd() + '/' + root_dir
    path = '/'.join([root_dir, 'results', device, model_name,
                     exp_type, experiment_name, ''])
    report_filename = 'REPORT_' + experiment_name + '.csv'
    data_filename = experiment_name + '_iter_' + str(iteration) + '.csv'

    print('Report saved at: ', path)

    if not os.path.exists(path):
        os.makedirs(path)

    if report_filename in os.listdir(path):
        report = pd.read_csv(path + report_filename)
    else:
        cols = ['recall', 'f1', 'precision',
                'accuracy', 'MAE', 'RETE', 'epochs', 'hparams']
        report = pd.DataFrame(columns=cols)
    hparams = {'hparams': model_hparams, 'epochs': int(epochs) + 1}
    report = report.append({**results, **hparams}, ignore_index=True)
    report.fillna(np.nan, inplace=True)
    report.to_csv(path + report_filename, index=False)

    if save_timeseries:
        cols = ['ground', 'preds']
        res_data = pd.DataFrame(list(zip(ground, preds)),
                            columns=cols)
        res_data.to_csv(path + data_filename, index=False)

    if plots:
        exp_list = experiment_name.split('_')
        device = exp_list[0]
        bounds = os.getcwd()+'/modules/plot_bounds/{}/{}_bounds_{}.csv'.format(exp_volume,exp_list[0],exp_list[1])
        bounds = pd.read_csv(str(bounds))
        bounds = bounds[(bounds['test_set'] == exp_list[6])&(bounds['test_house'] == int(exp_list[5]))]

        if not bounds.empty:
            low_lim = bounds['low_lim'].values[0]
            upper_lim = bounds['upper_lim'].values[0]

            display_res(root_dir, model_name, device, exp_type, experiment_name,
                        iteration, low_lim=low_lim, upper_lim=upper_lim,
                        plt_show=True, save_fig=True, save_dir='plots'
                       )
        else:
            print('Can"t plot, no experiment with name: {}'.format(experiment_name))

def display_res(root_dir=None, model_name=None, device=None,
                exp_type=None, experiment_name=None, iteration=None,
                low_lim=None, upper_lim=None, save_fig=True, plt_show=True, save_dir='plots'):
    if low_lim > upper_lim:
        low_lim, upper_lim = upper_lim, low_lim

    root_dir = root_dir
    path = '/'.join([root_dir, 'results', device, model_name,
                     exp_type, experiment_name,''])

    if os.path.exists(path):
        report_filename = 'REPORT_' + experiment_name + '.csv'
        snapshot_name = model_name + '_' + experiment_name + '_iter_' + str(iteration) + '.png'
        data_filename = experiment_name + '_iter_' + str(iteration) + '.csv'

        report = pd.read_csv(path + report_filename)

        # uncomment if wanna see the report
        # if int(iteration) > 0:
        #     print(report.iloc[int(iteration) - 1:int(iteration)])
        # else:
        #     print(report.iloc[int(iteration)])

        data = pd.read_csv(path + data_filename)
        data['ground'][low_lim:upper_lim].plot.line(legend=False,
                                                    # linestyle='dashed',
                                                   )
        data['preds'][low_lim:upper_lim].plot.line(legend=False)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.rcParams["figure.figsize"] = (8, 10)
        plt.legend(['ground truth', model_name])

        if save_fig:
            if save_dir:
                plt.savefig(root_dir+'/'+save_dir+'/'+snapshot_name,
                            #dpi=1000
                            )
            else:
                plt.savefig(path+snapshot_name)
        plt.clf()
        del ax

def get_tree_paths(tree_levels={}):
    tree_gen = (level for level in tree_levels)
    level = next(tree_gen)
    end = False
    if level == 'root':
        root_path = os.getcwd() + '/' + tree_levels[level]
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
                        paths.append(path)
            base_paths = paths
        except:
            end = True
    return base_paths

def get_exp_paths(cat_paths):
    exp_paths = []
    for cat_path in cat_paths:
        for exp in os.listdir(cat_path):
            exp_path = '/'.join([cat_path, exp])
            if os.path.exists(exp_path):
                exp_paths.append(exp_path)
    return exp_paths

def get_final_report(tree_levels, save=True, root_dir=None, save_name=None):

    path = '/'.join([root_dir, 'results', ''])
    columns = ['model', 'appliance','category','experiment',
           'recall','f1','precision','accuracy','MAE',
           'RETE','epochs','hparams']
    data = pd.DataFrame(columns=columns)

    cat_paths = get_tree_paths(tree_levels=tree_levels)
    exp_paths = get_exp_paths(cat_paths)

    for exp_path in exp_paths:
        for item in os.listdir(exp_path):
            if 'REPORT' in item:
                report = pd.read_csv(exp_path+'/'+item)
                model = exp_path.split('/')[-3]
                appliance = exp_path.split('/')[-1].split('_')[0]
                category = exp_path.split('/')[-2]
                experiment = exp_path.split('/')[-1]
                report['appliance'] = appliance
                report['model'] = model
                report['category'] = category
                report['experiment'] = experiment
                data = data.append(report, ignore_index=True)

    data = data[columns]
    data = data.sort_values(by=['appliance', 'experiment'])
    if save:
        data.to_csv(path+ save_name +'.csv',index=False)
    return data

def train_model(model_name, train_loader, test_loader,
                epochs=5, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
    """
    trainer = pl.Trainer(gpus=1, max_epochs=epochs)
    model = TrainingToolsFactory.build_and_equip_model(model_name=model_name, **kwargs)
    trainer.fit(model, train_loader)

    test_result = trainer.test(model, test_dataloaders=test_loader)
    metrics = test_result[0]['metrics']
    preds = test_result[0]['preds']

    return model, metrics, preds

def train_eval(model_name, train_loader, exp_type, tests_params,
               sample_period, batch_size, experiment_name, exp_volume,
               iteration, device, mmax, means, stds, meter_means, meter_stds,
               window_size, root_dir, data_dir, model_hparams,plots=True,save_timeseries=True,
               epochs=5, callbacks=None, val_loader=None,rolling_window=True,**kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run.
            It's used to look up the class in "model_dict"
    """
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks)
    model = TrainingToolsFactory.build_and_equip_model(model_name=model_name, model_hparams=model_hparams, **kwargs)
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    epochs = trainer.early_stopping_callback.stopped_epoch

    for i in range(len(tests_params)):
        building = tests_params['test_house'][i]
        dataset = tests_params['test_set'][i]
        dates = tests_params['test_date'][i]
        print(80 * '#')
        print('Evaluate house {} of {} for {}'.format(building, dataset, dates))
        print(80 * '#')

        datasource = DatasourceFactory.create_datasource(dataset)
        test_dataset = ElectricityDataset(datasource=datasource, building=int(building),
                                          window_size=window_size, device=device,
                                          dates=dates, mmax=mmax, means=means, stds=stds,
                                          meter_means=meter_means, meter_stds=meter_stds,
                                          sample_period=sample_period, rolling_window=rolling_window)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8)

        if rolling_window:
            ground = test_dataset.meterchunk.numpy()
        else:
            ground = test_dataset.meterchunk.numpy()
            ground = np.reshape(ground,-1)
        model.set_ground(ground)

        trainer.test(model, test_dataloaders=test_loader)
        test_result = model.get_res()
        results = test_result['metrics']
        preds = test_result['preds']
        final_experiment_name = experiment_name + 'test_' + building + '_' + dataset
        save_report(root_dir, model_name, device, exp_type, final_experiment_name, exp_volume,
                    iteration, results, preds, ground, model_hparams, epochs, plots=plots,save_timeseries=save_timeseries)
        del test_dataset, test_loader, ground, final_experiment_name

def create_timeframes(start, end, freq):
    '''
    freq(str): 'M' for month, 'D' for day
    start/end(str): the dates we want
        formats:
            '%Y-%m-%d' for 'D'
                or
            '%Y-%m' for 'M'
    when freq 'D', the dates are inclusive
    when freq 'M', the end date is exclusive
    '''
    # check if start <end else error
    datelist = pd.date_range(start, end, freq=freq).tolist()
    if freq=='D':
        date_format='%Y-%m-%d'
    else:
        date_format='%Y-%m'
    return [d.strftime(date_format) for d in datelist]

def create_time_folds(start_date, end_date, folds, freq='D', drop_last=False):
    '''
    receives a start and stop date and returns a dictionary
    with the necessary folds for train & test
    drop_last(bool): drops last dates to have folds with same lengths
    '''

    date_list = create_timeframes(start=start_date, end=end_date, freq='D')

    fold_len = len(date_list) // folds
    rest = len(date_list)-fold_len*folds
    print('#'*40)
    print('Folding for dates from {} to {}'.format(start_date, end_date))
    print('Total Number of days: ', len(date_list))
    print('Number of folds: ', folds)
    print('Length of each fold: ', fold_len)
    if drop_last:
        print('The last {} dates are dropped'.format(rest))
    else:
        print('Last fold has {} dates more'.format(rest))
    print('#'*40)

    date_folds = []
    for j in range(0, folds):
        if drop_last:
            date_folds.append(date_list[fold_len*(j):fold_len*(j+1)])
        else:
            if j<folds-1:
                date_folds.append(date_list[fold_len*(j):fold_len*(j+1)])
            else:
                date_folds.append(date_list[fold_len*(j):])

    date_bounds = [[day[0], day[-1]] for day in date_folds]

    final_folds = {}
    for fold in range(0, folds):
        test_dates = date_bounds[fold]
        train_1 = date_bounds[:fold]
        train_2 = date_bounds[fold+1:]
        if len(train_1):
            train_1 = [train_1[0][0],train_1[len(train_1)-1][-1]]
        if len(train_2):
            train_2 = [train_2[0][0],train_2[len(train_2)-1][-1]]

        final_folds[fold] = {'test_dates': test_dates, 'train_dates': [train_1, train_2]}

    return final_folds
