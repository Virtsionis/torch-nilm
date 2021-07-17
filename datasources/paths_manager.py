import os
from pathlib import Path

dirname = os.path.dirname(__file__)
UK_DALE = os.path.join(dirname, '../../Datasets/UKDALE/UKDALE.h5')
REDD = os.path.join(dirname, '../../Datasets/REDD/redd.h5')
REFIT = os.path.join(dirname, '../../Datasets/REFIT/REFIT.h5')


MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    return get_project_root().joinpath("data")


def get_results_path() -> Path:
    return get_project_root().joinpath("results")


def get_checkpoint_path() -> Path:
    return get_results_path().joinpath(MODEL_CKPT)
