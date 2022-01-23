import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants.constants import *
from constants.enumerates import DataTypes


def create_tree_dir(tree_levels: dict = None, clean: bool = False, plots: bool = False,
                    output_dir: str = DIR_OUTPUT_NAME):
    tree_gen = (level for level in tree_levels)
    level = next(tree_gen)
    end = False
    if output_dir:
        output_path = '/'.join([os.getcwd(), output_dir])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    else:
        output_path = os.getcwd()
    if level == ROOT_LEVEL:
        root_path = '/'.join([output_path, tree_levels[level]])
        if clean and os.path.exists(root_path):
            shutil.rmtree(root_path)
            print('all clean')
        if not os.path.exists(root_path):
            os.mkdir(root_path)

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
            end = True

        if plots:
            plot_path = '/'.join([root_path, tree_levels[EXPERIMENTS_LEVEL][0], DIR_PLOTS_NAME])
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)


def get_tree_paths(tree_levels: dict = None, output_dir: str = None):
    tree_gen = (level for level in tree_levels)
    level = next(tree_gen)
    end = False
    if level == ROOT_LEVEL and output_dir:
        root_path = os.getcwd() + '/' + output_dir + '/' + tree_levels[level]
    elif level == ROOT_LEVEL:
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


def get_exp_paths(cat_paths: list):
    exp_paths = []
    for cat_path in cat_paths:
        for exp in os.listdir(cat_path):
            exp_path = '/'.join([cat_path, exp])
            if os.path.exists(exp_path):
                exp_paths.append(exp_path)
    return exp_paths


def create_timeframes(start: object, end: object, freq: object):
    """
    freq(str): 'M' for month, 'D' for day
    start/end(str): the dates we want
        formats:
            '%Y-%m-%d' for 'D'
                or
            '%Y-%m' for 'M'
    when freq 'D', the dates are inclusive
    when freq 'M', the end date is exclusive
    """
    # check if start <end else error
    datelist = pd.date_range(start, end, freq=freq).tolist()
    if freq == 'D':
        date_format = '%Y-%m-%d'
    else:
        date_format = '%Y-%m'
    return [d.strftime(date_format) for d in datelist]


def create_time_folds(start_date: str, end_date: str, folds: int, drop_last: bool = False):
    """
    receives a start and stop date and returns a dictionary
    with the necessary folds for train & test
    drop_last(bool): drops last dates to have folds with same lengths
    """

    date_list = create_timeframes(start=start_date, end=end_date, freq='D')

    fold_len = len(date_list) // folds
    rest = len(date_list) - fold_len * folds
    print('#' * 40)
    print('Folding for dates from {} to {}'.format(start_date, end_date))
    print('Total Number of days: ', len(date_list))
    print('Number of folds: ', folds)
    print('Length of each fold: ', fold_len)
    if drop_last:
        print('The last {} dates are dropped'.format(rest))
    else:
        print('Last fold has {} dates more'.format(rest))
    print('#' * 40)

    date_folds = []
    for j in range(0, folds):
        if drop_last:
            date_folds.append(date_list[fold_len * (j):fold_len * (j + 1)])
        else:
            if j < folds - 1:
                date_folds.append(date_list[fold_len * (j):fold_len * (j + 1)])
            else:
                date_folds.append(date_list[fold_len * (j):])

    date_bounds = [[day[0], day[-1]] for day in date_folds]

    final_folds = {}
    for fold in range(0, folds):
        test_dates = date_bounds[fold]
        train_1 = date_bounds[:fold]
        train_2 = date_bounds[fold + 1:]
        if len(train_1):
            train_1 = [train_1[0][0], train_1[len(train_1) - 1][-1]]
        if len(train_2):
            train_2 = [train_2[0][0], train_2[len(train_2) - 1][-1]]

        final_folds[fold] = {TEST_DATES: test_dates, TRAIN_DATES: [train_1, train_2]}

    return final_folds


def rename_columns_by_type(data: pd.DataFrame, col_type: str, postfix: str):
    """
    This method renames all columns of a pandas DataFrame by a specified type adding a postfix
    at the end. After the renaming, returns the new dataframe.

    Args:
        data(pandas DataFrame): the target dataframe
        col_type(str): the type of the columns we want to rename
            'numeric'=> int64 or float64 type column
            'object' => string type column
        postfix(str): the string we want to add in the end of column names to be renamed
    """
    if col_type == NUMERIC_TYPE:
        rename_cols = data.select_dtypes(include=[DataTypes.INT64.value, DataTypes.FLOAT64.value]).columns.tolist()
    elif col_type == OBJECT_TYPE:
        rename_cols = data.select_dtypes(include=[DataTypes.OBJECT.value]).columns.tolist()
    else:
        rename_cols = data.select_dtypes(include=[col_type]).columns.tolist()

    rename_cols = {col: col + '_{}'.format(postfix) for col in rename_cols}
    data.rename(columns=rename_cols, inplace=True)
    return data


def pd_mean(data: pd.DataFrame, reset_index: bool = True):
    if reset_index:
        return data.mean().reset_index()
    return data.mean()


def pd_median(data: pd.DataFrame, reset_index: bool = True):
    if reset_index:
        return data.median().reset_index()
    return data.median()


def pd_std(data: pd.DataFrame, reset_index: bool = True):
    if reset_index:
        return data.std().reset_index()
    return data.std()


def pd_min(data: pd.DataFrame, reset_index: bool = True):
    if reset_index:
        return data.min().reset_index()
    return data.min()


def pd_max(data: pd.DataFrame, reset_index: bool = True):
    if reset_index:
        return data.max().reset_index()
    return data.max()


def pd_quantile(data: pd.DataFrame, q: float = .1, reset_index: bool = True):
    if reset_index:
        return data.quantile(q).reset_index()
    return data.quantile(q)


def quantile_25(data: pd.DataFrame, reset_index: bool = True):
    return pd_quantile(data, q=.25, reset_index=reset_index)


def quantile_75(data: pd.DataFrame, reset_index: bool = True):
    return pd_quantile(data, q=.75, reset_index=reset_index)


def destandardize(data: np.array, means: float, stds: float):
    return (data * stds) + means


def denormalize(data: np.array, mmax: float):
    return data * mmax


def experiment_name_format(x):
    x = x.split('_')[1:]
    return ' '.join(x)


def list_intersection(l1, l2):
    if not l2 and len(l1):
        return l1
    if not l1 and len(l2):
        return l2
    if l1 and l2:
        return list(set(l1) & set(l2))
