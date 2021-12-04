import warnings
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
    return list(STATISTIC_MEASURES.keys())


def get_statistical_report(save_name=None, data=None, data_filename=None, root_dir=None, stat_measures=[]):
    """
    Method that re-formats the report from the generate report to an excel-type report with statistical calculations.
    The data can either be loaded from disk or given as pandas DataFrame.

    save_name(str): the name of the result report xlsx file, without the '.xlsx'
    data(pandas DataFrame): the data generated from the 'generate report' method
    data_filename(str): the name of the report data generated from the 'generate report' method, if user wants to load
        from disk
    root_dir(str): the root folder of the project
    stat_measures(list of strings): user can define the appropriate statistical measures to be included to the report
        e.g:  stat_measures = ['mean', 'std', 'max', 'min']
    """

    if root_dir:
        data_path = '/'.join([root_dir, 'results', ''])
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
        output_path = '{}final_report.xlsx'.format(data_path)

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
                x = rename_columns_by_type(x, 'numeric', measure)
                results.append(x)
            else:
                stat_measures.remove(measure)
                warnings.warn('Unsupported requested measure with name: {}, excluded from the report.'.format(measure))

        df = reduce(lambda left, right: pd.merge(left, right, on=categorical_cols), results)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols.sort(reverse=False)
        df = df[categorical_cols + numeric_cols]

        with pd.ExcelWriter(output_path) as writer:
            for appliance in df['appliance'].unique():
                temp = df[df['appliance'] == appliance]
                temp = temp.sort_values(by=['category', 'experiment'])
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
    """
    if metrics:
        columns = ['model', 'appliance', 'category', 'experiment'] + metrics + ['epochs', 'hparams']
    else:
        columns = ['model', 'appliance', 'category', 'experiment',
                   'recall', 'f1', 'precision', 'accuracy', 'MAE',
                   'RETE', 'epochs', 'hparams']

    path = '/'.join([root_dir, 'results', ''])
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
                report['appliance'] = appliance
                report['model'] = model
                report['category'] = category
                report['experiment'] = experiment
                data = data.append(report, ignore_index=True, sort=False)

    data = data[columns]
    data = data.sort_values(by=['appliance', 'experiment'])
    data['epochs'] = data['epochs'].astype('int')
    if save:
        data.to_csv(path + save_name + '.csv', index=False)
    return data
