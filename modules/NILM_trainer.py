import numpy as np
import pytorch_lightning as pl
from constants.constants import*
from torch.utils.data import DataLoader
from lab.training_tools import TrainingToolsFactory
from modules.reporting import save_appliance_report
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityIterableDataset, ElectricityDataset


def train_eval(model_name: str, train_loader: DataLoader, exp_type: str, tests_params: list, sample_period: int,
               batch_size: int, experiment_name: str, exp_volume: str, iteration: int, device: str, mmax: float,
               means: float, stds: float, meter_means: float, meter_stds: float, window_size: int, root_dir: str,
               data_dir: str, model_hparams: dict, plots: bool = True, save_timeseries: bool = True, epochs: int = 5,
               callbacks=None, val_loader: DataLoader = None, rolling_window: bool = True, inference_cpu: bool = True,
               **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run.
            It's used to look up the class in "model_dict"
    """
    progress_bar = True
    if progress_bar:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks)
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks,
                             progress_bar_refresh_rate=0)

    model = TrainingToolsFactory.build_and_equip_model(model_name=model_name, model_hparams=model_hparams, **kwargs)
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    epochs = trainer.early_stopping_callback.stopped_epoch

    for i in range(len(tests_params)):
        building = tests_params[TEST_HOUSE][i]
        dataset = tests_params[TEST_SET][i]
        dates = tests_params[TEST_DATE][i]
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
            ground = np.reshape(ground, -1)
        if inference_cpu:
            print('Model to CPU')
            model.to('cpu')
        model.set_ground(ground)

        trainer.test(model, test_dataloaders=test_loader)
        test_result = model.get_res()
        results = test_result[COLUMN_METRICS]
        preds = test_result[COLUMN_PREDICTIONS]
        final_experiment_name = experiment_name + TEST_ID + building + '_' + dataset
        save_appliance_report(root_dir, model_name, device, exp_type, save_timeseries, final_experiment_name, exp_volume,
                              iteration, results, preds, ground, model_hparams, epochs, plots=plots)
        del test_dataset, test_loader, ground, final_experiment_name
