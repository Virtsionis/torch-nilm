import warnings

import numpy as np
from skimage.restoration import denoise_wavelet


def apply_rolling_window(mainchunk: np.array, meterchunk: np.array, window_size: int):
    if not window_size:
        raise Warning('Window size is not defined.')
    indexer = np.arange(window_size)[None, :] + np.arange(len(mainchunk) - window_size + 1)[:, None]
    mainchunk = mainchunk[indexer]
    meterchunk = meterchunk[window_size - 1:]
    return mainchunk, meterchunk


def apply_midpoint_window(mainchunk: np.array, meterchunk: np.array, window_size: int):
    if not window_size:
        raise Warning('Window size is not defined.')
    indexer = np.arange(window_size)[None, :] + np.arange(len(mainchunk) - window_size + 1)[:, None]
    mainchunk = mainchunk[indexer]
    midpoint = window_size // 2
    meterchunk = meterchunk[midpoint: len(mainchunk) + midpoint]
    return mainchunk, meterchunk


def apply_sequence_to_subsequence(mainchunk: np.array, meterchunk: np.array, sequence_window: int,
                                  subsequence_window: int):
    if not sequence_window:
        raise Warning('Sequence window is not defined.')
    if not subsequence_window:
        warnings.warn('Sub sequence window is not defined. So the 20% of sequence window was used.')
        subsequence_window = int(sequence_window * 0.2)
    upper_limit = (sequence_window + subsequence_window) // 2
    lower_limit = (sequence_window - subsequence_window) // 2
    sequence_indexer = np.arange(sequence_window)[None, :] + np.arange(len(mainchunk) - sequence_window + 1)[:, None]
    mainchunk = mainchunk[sequence_indexer]

    subsequence_indexer = np.arange(sequence_window)[lower_limit: upper_limit] + np.arange(len(mainchunk))[:, None]
    meterchunk = meterchunk[subsequence_indexer]
    return mainchunk, meterchunk


def apply_sequence_to_sequence(mainchunk: np.array, meterchunk: np.array, sequence_window: int):
    if not sequence_window:
        raise Warning('Sequence window is not defined.')
    sequence_indexer = np.arange(sequence_window)[None, :] + np.arange(len(mainchunk) - sequence_window + 1)[:, None]
    mainchunk = mainchunk[sequence_indexer]
    meterchunk = meterchunk[sequence_indexer]
    return mainchunk, meterchunk


def create_batches(mainchunk: np.array, meterchunk: np.array, seq_len: int):
    ix = mainchunk.index
    additional = seq_len - (len(ix) % seq_len)
    mainchunk = np.append(mainchunk, np.zeros(additional))
    meterchunk = np.append(meterchunk, np.zeros(additional))
    mainchunk = np.reshape(mainchunk, (int(len(mainchunk) / seq_len), seq_len, 1))
    meterchunk = np.reshape(meterchunk, (int(len(meterchunk) / seq_len), seq_len, 1))
    mainchunk = np.transpose(mainchunk, (0, 2, 1))
    meterchunk = np.transpose(meterchunk, (0, 2, 1))
    return mainchunk, meterchunk


def replace_nans(mainchunk: np.array, meterchunk: np.array):
    mainchunk.fillna(0, inplace=True)
    meterchunk.fillna(0, inplace=True)
    return mainchunk, meterchunk


def replace_nans_interpolation(mainchunk: np.array, meterchunk: np.array):
    mainchunk.interpolate(method='linear', limit_direction='forward', inplace=True)
    meterchunk.interpolate(method='linear', limit_direction='forward', inplace=True)
    return mainchunk, meterchunk


def normalize_chunks(mainchunk: np.array, meterchunk: np.array, mmax: float):
    if mmax is None:
        mmax = mainchunk.max()
    mainchunk = mainchunk / mmax
    meterchunk = meterchunk / mmax
    return mainchunk, meterchunk


def standardize_chunks(mainchunk: np.array, meterchunk: np.array, mains_mean: float,
                       mains_std: float, meter_mean: float, meter_std: float):
    if mains_mean is None and mains_std is None:
        mains_mean = mainchunk.mean()
        mains_std = mainchunk.std()

    if meter_mean is None and meter_std is None:
        meter_mean = meterchunk.mean()
        meter_std = meterchunk.std()

    mainchunk = (mainchunk - mains_mean) / mains_std
    meterchunk = (meterchunk - meter_mean) / meter_std
    return mainchunk, meterchunk


def is_bad_chunk(chunk: np.array):
    return (chunk == 0).all()


def align_chunks(mainchunk: np.array, meterchunk: np.array):
    mainchunk = mainchunk[~mainchunk.index.duplicated()]
    meterchunk = meterchunk[~meterchunk.index.duplicated()]
    ix = mainchunk.index.intersection(meterchunk.index)
    mainchunk = mainchunk[ix]
    meterchunk = meterchunk[ix]
    return mainchunk, meterchunk


def replace_with_zero_small_values(mainchunk: np.array, meterchunk: np.array, threshold: int):
    mainchunk[mainchunk < threshold] = 0
    meterchunk[meterchunk < threshold] = 0
    return mainchunk, meterchunk


def denoise(mainchunk: np.array, meterchunk: np.array):
    mainchunk = denoise_wavelet(mainchunk, wavelet='haar', wavelet_levels=3)
    meterchunk = denoise_wavelet(meterchunk, wavelet='haar', wavelet_levels=3)
    return mainchunk, meterchunk


def add_gaussian_noise(mainchunk: np.array, noise_factor: float = 0.1):
    noise = noise_factor * np.random.normal(0, 1, mainchunk.shape)
    mainchunk = mainchunk + noise
    return mainchunk
