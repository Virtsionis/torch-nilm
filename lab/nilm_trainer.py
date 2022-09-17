import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl
from constants.constants import*
from torch.utils.data import DataLoader
from lab.training_tools import TrainingToolsFactory
from utils.nilm_reporting import save_appliance_report
from datasources.datasource import DatasourceFactory
from datasources.torchdataset import ElectricityDataset, BaseElectricityMultiDataset
from constants.enumerates import SupportedPreprocessingMethods, SupportedFillingMethods
from pytorch_lightning.loggers import WandbLogger


def train_eval(model_name: str, train_loader: DataLoader, tests_params: pd.DataFrame, sample_period: int,
               batch_size: int, experiment_name: str, iteration: int, device: str, mmax: float,
               means: float, stds: float, meter_means: float, meter_stds: float, window_size: int, root_dir: str,
               model_hparams: dict, eval_params: dict, save_timeseries: bool = True, epochs: int = 5, callbacks=None,
               val_loader: DataLoader = None, preprocessing_method: str = SupportedPreprocessingMethods.ROLLING_WINDOW,
               fillna_method: str = SupportedFillingMethods.FILL_ZEROS, inference_cpu: bool = False,
               experiment_type: str = None, experiment_category: str = None, subseq_window: int = None,
               save_model: bool = False, saved_models_dir: str = DIR_SAVED_MODELS_NAME, model_index: int = None,
               save_preprocessing_params: bool = True, output_dir: str = DIR_OUTPUT_NAME, progress_bar: bool = True, ):
    """
    Inputs:
        model_name - Name of the model you want to run.
            It's used to look up the class in "model_dict"
    """

    if progress_bar:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks, precision=16,)
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks,
                             progress_bar_refresh_rate=0, precision=16)

    model = TrainingToolsFactory.build_and_equip_model(model_name=model_name,
                                                       model_hparams=model_hparams,
                                                       eval_params=eval_params)
    # print(model_hparams)
    # s = '/mnt/B40864F10864B450/WorkSpace/PHD/PHD_exps/torch_nilm/output/test2/benchmark/saved_models/washing machine/SAED/Single/washing machine_Single_Train_UKDALE_/SAED_iteration_1_16-May-2022-20:52:07.ckpt'
    # model = model.load_from_checkpoint(checkpoint_path=s)
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    epochs = trainer.early_stopping_callback.stopped_epoch

    if save_model:
        if output_dir:
            save_dir = '/'.join([os.getcwd(), output_dir, root_dir])
        else:
            save_dir = '/'.join([os.getcwd(), root_dir])

        model_path = '/'.join([save_dir, experiment_type, saved_models_dir, device, model_name,
                               experiment_category, experiment_name, ''])
        date = str(datetime.now().strftime("%d-%b-%Y-%H:%M:%S"))
        if model_index:
            filename = model_path + model_name + '_' + VERSION + '_' + str(model_index) + '_' + ITERATION_NAME + '_' + \
                       str(iteration) + '_' + date + CKPT_EXTENSION
        else:
            filename = model_path + model_name + '_' + ITERATION_NAME + '_' + str(iteration) + '_' + date +\
                       CKPT_EXTENSION
        trainer.save_checkpoint(filename)
        print('Model saved at: ', filename)

        if save_preprocessing_params:
            prepro_save_path = '/'.join([save_dir, experiment_type, saved_models_dir, device, ''])
            filename = prepro_save_path + PREPROCESSING_PARAMS_NAME + CSV_EXTENSION
            preprocessing_params = pd.DataFrame({COLUMN_MMAX: [mmax],
                                                 COLUMN_MEANS: [means],
                                                 COLUMN_STDS: [stds], })
            preprocessing_params.to_csv(filename, index=False)
            print('Preprocessing parameters saved at: ', filename)

    for i in range(len(tests_params)):
        building = tests_params[TEST_HOUSE][i]
        dataset = tests_params[TEST_SET][i]
        dates = tests_params[TEST_DATE][i]
        print(80 * '#')
        print('Evaluate house {} of {} for {}'.format(building, dataset, dates))
        print(80 * '#')

        datasource = DatasourceFactory.create_datasource(dataset)

        test_dataset = ElectricityDataset(datasource=datasource,
                                          building=int(building),
                                          window_size=window_size, subseq_window=subseq_window,
                                          device=device, dates=dates, mmax=mmax, means=means, stds=stds,
                                          meter_means=meter_means, meter_stds=meter_stds,
                                          sample_period=sample_period,
                                          preprocessing_method=preprocessing_method,
                                          fillna_method=fillna_method,)

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8)

        if preprocessing_method in [SupportedPreprocessingMethods.ROLLING_WINDOW,
                                    SupportedPreprocessingMethods.MIDPOINT_WINDOW]:
            ground = test_dataset.meterchunk.numpy()
        else:
            ground = test_dataset.meterchunk.numpy()
            ground = np.reshape(ground, -1)
        if inference_cpu:
            print('Model to CPU')
            model.to(CPU_NAME)
        model.set_ground(ground)

        trainer.test(model, test_dataloaders=test_loader)
        print('TESTING FINALISED')
        model_results = model.get_res()
        print('MODEL RESULTS', model_results)
        final_experiment_name = experiment_name + TEST_ID + building + '_' + dataset

        print('save report')
        save_appliance_report(root_dir=root_dir, model_name=model_name, device=device,
                              experiment_type=experiment_type, experiment_category=experiment_category,
                              save_timeseries=save_timeseries, experiment_name=final_experiment_name,
                              iteration=iteration, model_results=model_results, model_hparams=model_hparams,
                              epochs=epochs, model_index=model_index)
        print('REPORT SAVED')
        del test_dataset, test_loader, ground, final_experiment_name


def train_eval_super(model_name: str, train_loader: DataLoader, tests_params: pd.DataFrame, sample_period: int,
                     batch_size: int, experiment_name: str, iteration: int, devices: list, mmax: float,
                     means: float, stds: float, meter_means: float, meter_stds: float, window_size: int, root_dir: str,
                     model_hparams: dict, eval_params: dict, save_timeseries: bool = True, epochs: int = 5,
                     callbacks=None, val_loader: DataLoader = None,
                     preprocessing_method: str = SupportedPreprocessingMethods.SEQ_T0_SEQ,
                     fillna_method: str = SupportedFillingMethods.FILL_ZEROS, inference_cpu: bool = False,
                     experiment_type: str = None, experiment_category: str = None, subseq_window: int = None,
                     save_model: bool = False, saved_models_dir: str = DIR_SAVED_MODELS_NAME, model_index: int = None,
                     save_preprocessing_params: bool = True, output_dir: str = DIR_OUTPUT_NAME,
                     progress_bar: bool = True, ):
    """
    Inputs:
        model_name - Name of the model you want to run.
            It's used to look up the class in "model_dict"
    """

    wandb_logger = WandbLogger(name='SuperVAE1bc_{}'.format(str(iteration)), project='nikos')

    if progress_bar:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks, precision=16,
                             logger=wandb_logger)
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs, auto_lr_find=True, callbacks=callbacks, precision=16,
                             progress_bar_refresh_rate=0, logger=wandb_logger)

    model = TrainingToolsFactory.build_and_equip_model(model_name=model_name,
                                                       model_hparams=model_hparams,
                                                       eval_params=eval_params)
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
        test_dataset = BaseElectricityMultiDataset(datasource=datasource,
                                                   building=int(building),
                                                   window_size=window_size, subseq_window=subseq_window,
                                                   devices=devices,
                                                   start_date=dates[0],
                                                   end_date=dates[1],
                                                   mmax=mmax, means=means, stds=stds,
                                                   meter_means=meter_means, meter_stds=meter_stds,
                                                   sample_period=sample_period,
                                                   preprocessing_method=preprocessing_method, )

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8)

        if preprocessing_method in [SupportedPreprocessingMethods.ROLLING_WINDOW,
                                    SupportedPreprocessingMethods.MIDPOINT_WINDOW]:
            grounds = [meterchunk.numpy()for meterchunk in test_dataset.meterchunks]
        else:
            grounds = [np.reshape(meterchunk.numpy(), -1)for meterchunk in test_dataset.meterchunks]
        if inference_cpu:
            print('Model to CPU')
            model.to(CPU_NAME)
        model.set_ground(grounds)

        trainer.test(model, test_dataloaders=test_loader)
        print('TESTING FINALISED')
        model_resultss = model.get_res()

        for model_results in model_resultss:
            final_experiment_name = model_results[COLUMN_DEVICE] + '_' + experiment_name + TEST_ID + building + '_' + dataset

            save_appliance_report(root_dir=root_dir, model_name=model_name, device=model_results[COLUMN_DEVICE],
                                  experiment_type=experiment_type, experiment_category=experiment_category,
                                  save_timeseries=save_timeseries, experiment_name=final_experiment_name,
                                  iteration=iteration, model_results=model_results, model_hparams=model_hparams,
                                  epochs=epochs, model_index=model_index)
            print('REPORT SAVED')
#
#
# def compute_metrics(res):
#     def f1_score_from_stats(tp, fp, fn, average='micro'):
#         assert len(tp) == len(fp)
#         assert len(fp) == len(fn)
#
#         if average not in set(['micro', 'macro']):
#             raise ValueError("Specify micro or macro")
#
#         if average == 'micro':
#             f1 = 2 * np.sum(tp) / \
#                  float(2 * np.sum(tp) + np.sum(fp) + np.sum(fn))
#
#         elif average == 'macro':
#
#             def safe_div(a, b):
#                 """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     c = np.true_divide(a, b)
#                 return c[np.isfinite(c)]
#
#             f1 = np.mean(safe_div(2 * tp, 2 * tp + fp + fn))
#
#         return f1
#
#     mif1 = round(f1_score_from_stats(tp, fp, fn, average='micro'), 4)
#     maf1 = round(f1_score_from_stats(tp, fp, fn, average='macro'), 4)
#
#     return mif1, maf1




