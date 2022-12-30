from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasources import paths_manager


class TrainerCallbacksFactory:

    @staticmethod
    def create_earlystopping() -> EarlyStopping:
        return EarlyStopping(monitor='val_loss',
                             min_delta=1e-4,
                             patience=5,
                             verbose=True,
                             mode='min')

    @staticmethod
    def create_modelcheckpoint() -> ModelCheckpoint:
        return ModelCheckpoint(
            dirpath=paths_manager.get_checkpoint_path().__str__(),
            verbose=False,
            monitor='val_loss',
            mode='min'
        )


class LoggerCallbacksFactory:

    @staticmethod
    def create_wandblogger(name, project: str = 'ib-pool', job_type: str = 'train', offline=False) -> WandbLogger:
        return WandbLogger(name=name, project=project, job_type=job_type,
                           save_dir=paths_manager.get_results_path().__str__(),
                           offline=offline)