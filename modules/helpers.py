import os, shutil
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from modules.MyTrainer import NILMTrainer

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

def save_report(root_dir=None, model_name=None, device=None, experiment_name=None,
                iteration=None, results={}, preds=None, ground=None):

    root_dir = os.getcwd() + '/' + root_dir
    path = '/'.join([root_dir,'results',device,model_name,''])
    report_filename = 'REPORT_' + experiment_name + '.csv'
    data_filename = experiment_name + '_iter_' + str(iteration)+'.csv'

    if os.path.exists(path):
        if report_filename in os.listdir(path):
            report = pd.read_csv(path+report_filename)
        else:
            cols = ['recall','f1','precision'
                    'accuracy','MAE','RETE']
            report = pd.DataFrame(columns=cols)
        report = report.append(results, ignore_index=True)
        report.fillna(np.nan, inplace=True)
        report.to_csv(path+report_filename, index=False)

        cols = ['ground','preds']
        res_data = pd.DataFrame(list(zip(ground, preds)),
                                columns=cols)
        res_data.to_csv(path+data_filename, index=False)
    else:
        return 'ERROR, specified path does not exist'

def final_device_report():
    pass

def train_model(model_name, train_loader, test_loader,
                save_name=None, epochs=5, **kwargs):
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