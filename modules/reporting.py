import warnings
import numpy as np
from modules.helpers import*
from functools import reduce

STATISTIC_MEASURES = {
        'mean': pd_mean,
        'median': pd_median,
        'std': pd_std,
        'min': pd_min,
        'max': pd_max,
        '25th_percentile': quantile_25,
        '75th_percentile': quantile_75,
}


def get_supported_stat_measures():
    """
    returns the supported statistical measures
    """
    return list(STATISTIC_MEASURES.keys())


def get_statistical_report(save_name=None, data=None, data_filename=None, root_dir=None, stat_measures=[]):
    """
    Method that re-formats the report from get_final_report to an excel-type report with statistical calculations.
    The data can either be loaded from disk or given as pandas DataFrame.

    Args:
        save_name(str): the name of the result report xlsx file, without the '.xlsx'
        data(pandas DataFrame): the data generated from the 'generate report' method
        data_filename(str): the name of the report data generated from the 'generate report' method,
            if user wants to load from disk
        root_dir(str): the root folder of the project
        stat_measures(list of strings): user can define the appropriate statistical measures to be included to the report
            supported measures: ['mean', 'median', 'std', 'min', 'max', '25th_percentile', '75th_percentile']

    Example of use:
        report = get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='single_building_exp')
        get_statistical_report(save_name='test', data=None, data_filename='single_building_exp',
                       root_dir=ROOT, stat_measures=['min', '75th_percentile'])

                                            or

        get_statistical_report(save_name='test', data=report, data_filename=None,
                               root_dir=ROOT, stat_measures=['min', '75th_percentile'])
    """

    if root_dir:
        data_path = '/'.join([root_dir, DIR_RESULTS_NAME, ''])
    else:
        data_path = ''

    try:
        if data_filename and data is None:
            data = pd.read_csv(data_path + data_filename + '.csv')
    except Exception as e:
        raise Exception(e)

    if save_name:
        output_path = '{}{}.xlsx'.format(data_path, save_name)
    else:
        output_path = '{}{}.xlsx'.format(data_path, DEFAULT_FINAL_REPORT_NAME)

    if data.empty or data is None:
        raise Exception('Empty dataframe given, no report is generated.')
    else:
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        grouped_data = data.groupby(by=categorical_cols)

        if not stat_measures:
            stat_measures = ['mean', 'std']

        results = []
        for measure in stat_measures:
            if measure in STATISTIC_MEASURES.keys():
                x = STATISTIC_MEASURES[measure](grouped_data)
                x = rename_columns_by_type(x, NUMERIC_TYPE, measure)
                results.append(x)
            else:
                stat_measures.remove(measure)
                warnings.warn('Unsupported requested measure with name: {}, excluded from the report.'.format(measure))

        df = reduce(lambda left, right: pd.merge(left, right, on=categorical_cols), results)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols.sort(reverse=False)
        df = df[categorical_cols + numeric_cols]

        with pd.ExcelWriter(output_path) as writer:
            for appliance in df[COLUMN_APPLIANCE].unique():
                temp = df[df[COLUMN_APPLIANCE] == appliance]
                temp = temp.sort_values(by=[COLUMN_CATEGORY, COLUMN_EXPERIMENT])
                temp.to_excel(writer,
                              sheet_name=appliance.upper(),
                              engine='xlsxwriter',
                              freeze_panes=(1, 4),
                              index=False,
                              )


def get_final_report(tree_levels, save=True, root_dir=None, save_name=None, metrics=[]):
    """
    This method merges all produced reports in one csv file. To generate the
    report file, the tree structure of the resulted reports should be given.
    Args:
        tree_levels(dict): the tree structure of the project
        save(bool): variable that controls if the resulted file should be saved
            Default value is True
        root_dir(str): the ROOT of the project
        save_name(str): the name of the resulted file
        metrics(list of str): the metrics to be included in the report

    Example of use:
        dev_list = [
            'washing machine',
            'kettle',
        ]
        mod_list = [
            'SAED',
            'WGRU',
        ]
        ROOT = 'Params'
        cat_list = [x for x in ['Single', 'Multi']]
        tree_levels = {'root': ROOT, 'l1': ['results'], 'l2': dev_list, 'l3': mod_list, 'experiments': cat_list}
        report = get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='single_building_exp')
    """
    if metrics:
        columns = [COLUMN_MODEL, COLUMN_APPLIANCE, COLUMN_CATEGORY, COLUMN_EXPERIMENT] + metrics +\
                   [COLUMN_EPOCHS, COLUMN_HPARAMS]
    else:
        columns = [COLUMN_MODEL, COLUMN_APPLIANCE, COLUMN_CATEGORY, COLUMN_EXPERIMENT,
                   COLUMN_RECALL, COLUMN_F1, COLUMN_PRECISION, COLUMN_ACCURACY, COLUMN_MAE,
                   COLUMN_RETE, COLUMN_EPOCHS, COLUMN_HPARAMS]

    path = '/'.join([root_dir, DIR_RESULTS_NAME, ''])
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
                report[COLUMN_APPLIANCE] = appliance
                report[COLUMN_MODEL] = model
                report[COLUMN_CATEGORY] = category
                report[COLUMN_EXPERIMENT] = experiment
                data = data.append(report, ignore_index=True, sort=False)

    data = data[columns]
    data = data.sort_values(by=[COLUMN_APPLIANCE, COLUMN_EXPERIMENT])
    data[COLUMN_EPOCHS] = data[COLUMN_EPOCHS].astype('int')
    if save:
        data.to_csv(path + save_name + '.csv', index=False)
    return data


def save_appliance_report(root_dir=None, model_name=None, device=None, exp_type=None, save_timeseries=True,
                          experiment_name=None, exp_volume='large', iteration=None, results={},
                          preds=None, ground=None, model_hparams=None, epochs=None, plots=True):

    root_dir = os.getcwd() + '/' + root_dir
    path = '/'.join([root_dir, DIR_RESULTS_NAME, device, model_name,
                     exp_type, experiment_name, ''])
    report_filename = REPORT_PREFIX + experiment_name + '.csv'
    data_filename = experiment_name + ITERATION_ID + str(iteration) + '.csv'

    if not os.path.exists(path):
        os.makedirs(path)

    if report_filename in os.listdir(path):
        report = pd.read_csv(path + report_filename)
    else:
        cols = [COLUMN_RECALL, COLUMN_F1, COLUMN_PRECISION, COLUMN_ACCURACY,
                COLUMN_MAE, COLUMN_RETE, COLUMN_EPOCHS, COLUMN_HPARAMS]
        report = pd.DataFrame(columns=cols)
    hparams = {COLUMN_HPARAMS: model_hparams, COLUMN_EPOCHS: int(epochs) + 1}
    report = report.append({**results, **hparams}, ignore_index=True)
    report.fillna(np.nan, inplace=True)
    report.to_csv(path + report_filename, index=False)

    if save_timeseries:
        cols = [COLUMN_GROUNDTRUTH, COLUMN_PREDICTIONS]
        res_data = pd.DataFrame(list(zip(ground, preds)),
                                columns=cols)
        res_data.to_csv(path + data_filename, index=False)
        print('Time series saved at: ', path + data_filename)

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
            raise Exception('Can"t plot, no experiment with name: {}'.format(experiment_name))
