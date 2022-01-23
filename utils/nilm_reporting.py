import warnings
from functools import reduce
from utils.helpers import *
from utils.plotting import plot_dataframe
from constants.enumerates import StatMeasures

STATISTIC_MEASURES = {
    StatMeasures.MEAN: pd_mean,
    StatMeasures.MEDIAN: pd_median,
    StatMeasures.STANDARD_DEVIATION: pd_std,
    StatMeasures.MINIMUM: pd_min,
    StatMeasures.MAXIMUM: pd_max,
    StatMeasures.PERCENTILE_25TH: quantile_25,
    StatMeasures.PERCENTILE_75TH: quantile_75,
}


def get_supported_stat_measures():
    """
    returns the supported statistical measures
    """
    return [measure.name for measure in STATISTIC_MEASURES.keys()]


def get_statistical_report(save_name: str = None, data: pd.DataFrame = None, data_filename: str = None,
                           root_dir: str = None, output_dir: str = DIR_OUTPUT_NAME, stat_measures: list = None,
                           save_plots: bool = True, **plot_args):
    """
    Method that re-formats the report from get_final_report to an excel-type report with statistical calculations.
    The data can either be loaded from disk or given as pandas DataFrame.

    Args:
        save_name(str): the name of the result report xlsx file, without the '.xlsx'
        data(pandas DataFrame): the data generated from the 'generate report' method
        data_filename(str): the name of the report data generated from the 'generate report' method,
            if user wants to load from disk
        output_dir(str): the OUTPUT of where torch-nilm projects are put
        root_dir(str): the root folder of the project
        stat_measures(list of strings): user can define the appropriate statistical measures to be included to the report
            supported measures: ['mean', 'median', 'std', 'min', 'max', '25th_percentile', '75th_percentile']
        save_plots: whether to save bar plots and/or spider plots from the results
        plot_args: plot settings

    Example of use:
        report = get_final_report(tree_levels, save=True, root_dir=ROOT, save_name='single_building_exp')
        get_statistical_report(save_name='test', data=None, data_filename='single_building_exp',
                       root_dir=ROOT, stat_measures=['min', '75th_percentile'])

                                            or

        get_statistical_report(save_name='test', data=report, data_filename=None,
                               root_dir=ROOT, stat_measures=['min', '75th_percentile'])
    """
    if stat_measures is None:
        stat_measures = []
    if output_dir:
        data_path = '/'.join([output_dir, root_dir, DIR_RESULTS_NAME, ''])
        plots_path = '/'.join([output_dir, root_dir, DIR_PLOTS_NAME, ''])
    else:
        data_path = '/'.join([root_dir, DIR_RESULTS_NAME, ''])
        plots_path = '/'.join([root_dir, DIR_PLOTS_NAME, ''])

    try:
        if data_filename and data is None:
            data = pd.read_csv(data_path + data_filename + CSV_EXTENSION)
    except Exception as e:
        raise Exception(e)

    if save_name:
        output_path = '{}{}{}'.format(data_path, save_name, XLSX_EXTENSION)
    else:
        output_path = '{}{}{}'.format(data_path, DEFAULT_FINAL_REPORT_NAME, XLSX_EXTENSION)

    if data.empty or data is None:
        raise Exception('Empty dataframe given, no report is generated.')
    else:
        categorical_cols = data.select_dtypes(include=[DataTypes.OBJECT.value]).columns.tolist()
        grouped_data = data.groupby(by=categorical_cols)

        if not stat_measures:
            stat_measures = [StatMeasures.MEAN, StatMeasures.STANDARD_DEVIATION]

        results = []
        for measure in stat_measures:
            if measure in STATISTIC_MEASURES.keys():
                x = STATISTIC_MEASURES[measure](grouped_data)
                x = rename_columns_by_type(x, NUMERIC_TYPE, measure.value)
                results.append(x)
            else:
                stat_measures.remove(measure)
                warnings.warn('Unsupported requested measure with name: {}, excluded from the report.'.format(measure))

        df = reduce(lambda left, right: pd.merge(left, right, on=categorical_cols), results)

        numeric_cols = df.select_dtypes(include=[DataTypes.INT64.value, DataTypes.FLOAT64.value]).columns.tolist()
        numeric_cols.sort(reverse=False)
        df = df[categorical_cols + numeric_cols]

        with pd.ExcelWriter(output_path) as writer:
            for appliance in df[COLUMN_APPLIANCE].unique():
                temp = df[df[COLUMN_APPLIANCE] == appliance]
                temp = temp.sort_values(by=[COLUMN_CATEGORY, COLUMN_EXPERIMENT])
                temp.to_excel(writer,
                              sheet_name=appliance.upper(),
                              engine=XLSX_ENGINE,
                              freeze_panes=(1, 4),
                              index=False,
                              )
                if save_plots:
                    plot_dataframe(data=temp, plots_save_path=plots_path, **plot_args)


def get_final_report(tree_levels: dict, save: bool = True, root_dir: str = None, output_dir: str = DIR_OUTPUT_NAME,
                     save_name: str = None, metrics: list = None, model_index: int = None):
    """
    This method merges all produced reports in one csv file. To generate the
    report file, the tree structure of the resulted reports should be given.
    Args:
        tree_levels(dict): the tree structure of the project
        save(bool): variable that controls if the resulted file should be saved
            Default value is True
        output_dir(str): the OUTPUT of where torch-nilm projects are put
        root_dir(str): the ROOT of the project
        save_name(str): the name of the resulted file
        metrics(list of str): the metrics to be included in the report
        model_index(int): model versioning

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
        columns = [COLUMN_MODEL, COLUMN_APPLIANCE, COLUMN_CATEGORY, COLUMN_EXPERIMENT] + metrics\
                  + [COLUMN_EPOCHS, COLUMN_HPARAMS]
    else:
        columns = [COLUMN_MODEL, COLUMN_APPLIANCE, COLUMN_CATEGORY, COLUMN_EXPERIMENT,
                   COLUMN_RECALL, COLUMN_F1, COLUMN_PRECISION, COLUMN_ACCURACY, COLUMN_MAE,
                   COLUMN_RETE, COLUMN_EPOCHS, COLUMN_HPARAMS]

    if model_index:
        columns.append(COLUMN_MODEL_VERSION)

    if output_dir:
        path = '/'.join([output_dir, root_dir, DIR_RESULTS_NAME, ''])
    else:
        path = '/'.join([root_dir, DIR_RESULTS_NAME, ''])
    data = pd.DataFrame(columns=columns)

    cat_paths = get_tree_paths(tree_levels=tree_levels, output_dir=output_dir)
    exp_paths = get_exp_paths(cat_paths)
    for exp_path in exp_paths:
        for item in os.listdir(exp_path):
            if REPORT in item:
                report = pd.read_csv(exp_path + '/' + item)
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
    data[COLUMN_EPOCHS] = data[COLUMN_EPOCHS].astype(DataTypes.INT.value)
    if save:
        data.to_csv(path + save_name + CSV_EXTENSION, index=False)
    return data


def save_appliance_report(root_dir: str = None, model_name: str = None, device: str = None,
                          experiment_type: str = None, experiment_category: str = None, save_timeseries: bool = True,
                          experiment_name: str = None, iteration: int = None, model_results: dict = None,
                          model_hparams: dict = None, epochs: int = None, output_dir: str = DIR_OUTPUT_NAME,
                          model_index: int = None):
    if output_dir:
        root_dir = '/'.join([os.getcwd(), output_dir, root_dir])
    else:
        root_dir = '/'.join([os.getcwd(), root_dir])

    path = '/'.join([root_dir, experiment_type, DIR_RESULTS_NAME, device, model_name,
                     experiment_category, experiment_name, ''])
    report_filename = REPORT_PREFIX + experiment_name + CSV_EXTENSION

    hparams = {COLUMN_HPARAMS: model_hparams, COLUMN_EPOCHS: int(epochs) + 1}
    if model_index:
        data_filename = experiment_name + '_' + VERSION + '_' + str(model_index) + ITERATION_ID + str(iteration) + \
                        CSV_EXTENSION
        hparams[COLUMN_MODEL_VERSION] = VERSION + str(model_index)
    else:
        data_filename = experiment_name + ITERATION_ID + str(iteration) + CSV_EXTENSION

    if not os.path.exists(path):
        os.makedirs(path)

    if report_filename in os.listdir(path):
        report = pd.read_csv(path + report_filename)
    else:
        cols = [COLUMN_RECALL, COLUMN_F1, COLUMN_PRECISION, COLUMN_ACCURACY,
                COLUMN_MAE, COLUMN_RETE, COLUMN_EPOCHS, COLUMN_HPARAMS]
        report = pd.DataFrame(columns=cols)

    try:
        results = model_results[COLUMN_METRICS]
        preds = model_results[COLUMN_PREDICTIONS]
        ground = model_results[COLUMN_GROUNDTRUTH]
    except Exception as exception:
        raise exception

    report = report.append({**results, **hparams}, ignore_index=True)
    report.fillna(np.nan, inplace=True)
    report.to_csv(path + report_filename, index=False)
    print('Report saved at: ', path)

    if save_timeseries:
        cols = [COLUMN_GROUNDTRUTH, COLUMN_PREDICTIONS]
        res_data = pd.DataFrame(list(zip(ground, preds)),
                                columns=cols)
        res_data.to_csv(path + data_filename, index=False)
        print('Time series saved at: ', path + data_filename)
