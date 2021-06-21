from numpy.lib.npyio import save
import torch
import pandas as pd
from modules.helpers import create_tree_dir,train_eval
from torch.utils.data import DataLoader


from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset,ElectricityIterableDataset
from modules.MyDataSet import MyChunkList #,MyChunk

with torch.no_grad():
    torch.cuda.empty_cache()

clean = False
ROOT = 'ablation_study'#'output'
data_dir = '/mnt/B40864F10864B450/WorkSpace/PHD/PHD_exps/data'
train_file_dir = 'dates2/train/'
test_file_dir = 'dates2/test/'

dev_list = [
#             'dish washer',
            # 'microwave',
#             'washing machine',
            'kettle',
#             'tumble dryer',
#             'fridge',
#             'washer dryer',
#             'television',
#             'computer',
#             'electric space heater'
           ]
mod_list = [
#             'PAF'
            # 'PAFnet',
            'S2P',
            # 'SimpleGru',
            # 'SAED',
            # 'FFED',
            'WGRU',
            # 'FNET',
            # 'ConvFourier'
            ]
cat_list = [x for x in ['Single', 'Multi']]
tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments':cat_list}
create_tree_dir(tree_levels=tree_levels, clean=clean)

# Experiment Settings
exp_type = 'Single'#'Multi'

EPOCHS = 50
ITERATIONS = 3
SAVE = True
LOGGER = False

SAMPLE_PERIOD = 6

BATCH = 512
dropout = 0.5

windows = {
            'fridge': 50,
            'kettle': 50,
            'washing machine': 100,
            'washer dryer': 100,
            'tumble dryer': 100,
            'dish washer': 50,
            'microwave': 50,
            'television': 50,
            'computer': 50,
            'electric space heater': 100,
}
# Run

for device in dev_list:

    WINDOW = windows[device]
    model_hparams={
                  'SimpleGru':{},
                  'SAED':{'window_size':WINDOW},
                  'FFED':{},
                  'WGRU':{'dropout':dropout},
                  'S2P': {'window_size':WINDOW, 'dropout':dropout},
                  'ConvFourier': {'window_size':WINDOW, 'dropout':dropout},
                  'PAF': {'window_size':WINDOW, 'dropout':dropout},
                  'PAFnet': {'cnn_dim': 16, 'kernel_size':5, 'depth':4,
                             'window_size':WINDOW,'hidden_factor': 10, 'dropout':dropout},
                  'FNET': {'depth': 3, 'kernel_size':8, 'cnn_dim': 64,
                           'input_dim':WINDOW, 'hidden_dim':WINDOW*8, 'dropout':dropout},
                   }

    test_houses = []
    test_sets = []
    test_dates = []
    test_file = open('{}base{}TestSetsInfo_{}'.format(test_file_dir, exp_type, device), 'r')
    for line in test_file:
        toks = line.split(',')
        test_sets.append(toks[0])
        test_houses.append(toks[1])
        test_dates.append([str(toks[2]),str(toks[3].rstrip("\n"))])
    test_file.close()
    data = {'test_house': test_houses, 'test_set': test_sets, 'test_date': test_dates}
    tests_params = pd.DataFrame(data)
    print(device)
    print(tests_params)

    # TRAIN LOADER
    train_file = open('{}base{}TrainSetsInfo_{}'.format(train_file_dir, exp_type, device), 'r')
    if exp_type=='Single':
        for line in train_file:
            toks = line.split(',')
            train_set = toks[0]
            train_house = toks[1]
            train_dates = [str(toks[2]),str(toks[3].rstrip("\n"))]
            break
        train_file.close()
        print(train_dates)
        path = data_dir + '/{}/{}.h5'.format(train_set, train_set)
        print(path)
        path = data_dir + '/{}/{}.h5'.format(train_set, train_set)
        print(path)
        datasource = DatasourceFactory.create_datasource(train_set)
        train_dataset = ElectricityDataset(datasource=datasource, building=int(train_house), window_size=WINDOW,
                                device=device, dates=train_dates, sample_period=SAMPLE_PERIOD)
    else:
        for line in train_file:
            toks = line.split(',')
            train_set = toks[0]
            break
        train_file.close()
        train_dataset = MyChunkList(device,filename=train_file,
                                    window_size=WINDOW, sample_period=SAMPLE_PERIOD)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, 
                              shuffle=True, num_workers=8)
    mmax = train_dataset.mmax

    # RUN DEVICE EXPERIMENT FOR ALL MODELS
    experiments = []
    experiment_name = '_'.join([device, exp_type,'Train',train_set,'',])
    print(experiment_name)
    eval_params={'device':device,
                 'mmax':mmax,
                 'groundtruth':''}
    for model_name in mod_list:
        print('#'*40)
        print('MODEL: ', model_name)
        print('#'*40)
        for iteration in range(1,ITERATIONS+1):
            print('#'*20)
            print('Iteration: ', iteration)
            print('#'*20)
            experiment_name = '_'.join([device, exp_type,'Train',train_set,'',])
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
                       save=SAVE,
                       logger=LOGGER,
                       eval_params=eval_params,
                       model_hparams=model_hparams[model_name])
