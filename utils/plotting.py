import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from constants.constants import *
from constants.enumerates import StatMeasures
from utils.helpers import list_intersection, experiment_name_format

pd.set_option('mode.chained_assignment', None)


def plot_radar_chart(data: pd.DataFrame = None, num_columns: list = None, plot_title: str = None,
                     save_name: str = None, ):
    if data.empty:
        warnings.warn('Empty dataframe')
    else:
        fig = go.Figure()
        models = data[COLUMN_MODEL].unique()

        for col in num_columns:
            if COLUMN_MAE in col:
                data.loc[:, col] = 1 - data[col] / data[col].max()
            if COLUMN_RETE in col:
                data.loc[:, col] = 1 - data[col]

        for model in models:
            r = data[data[COLUMN_MODEL] == model][num_columns].values[0].tolist()
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=[n.upper() for n in num_columns],
                fill='toself',
                name=model
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=plot_title
        )

        fig.write_image(save_name)


def plot_dataframe(data: pd.DataFrame = None, metrics: list = None, measures: list = None, appliances: list = None,
                   categories: list = None, experiments: list = None, plot_bar: bool = True, plot_spider: bool = True,
                   plots_save_path: str = None,):

    if metrics is None:
        metrics = [COLUMN_F1, COLUMN_MAE, COLUMN_RETE, COLUMN_RECALL, COLUMN_PRECISION, COLUMN_ACCURACY]
    if measures is None:
        measures = [StatMeasures.MEAN]

    categories = list_intersection(data[COLUMN_CATEGORY].unique(), categories)
    experiments = list_intersection(data[COLUMN_EXPERIMENT].unique(), experiments)

    num_cols = ['_'.join([metric, measure.value]) for metric in metrics for measure in measures]
    num_cols = list_intersection(num_cols, data.columns.tolist())

    cat_cols = [COLUMN_MODEL, COLUMN_MODEL_VERSION, COLUMN_CATEGORY, COLUMN_APPLIANCE, COLUMN_EXPERIMENT]
    cat_cols = list_intersection(cat_cols, data.columns.tolist())

    num_cols.sort()
    cat_cols.sort()

    data = data[cat_cols + num_cols]
    data = data[(data[COLUMN_CATEGORY].isin(categories)) & (data[COLUMN_EXPERIMENT].isin(experiments))]

    if COLUMN_MODEL_VERSION in cat_cols:
        data[COLUMN_MODEL] = data[COLUMN_MODEL] + ' ' + data[COLUMN_MODEL_VERSION]

    if not appliances:
        appliances = data[COLUMN_APPLIANCE].unique()

    appliances = [app.title() for app in appliances]
    for appliance in appliances:
        for category in categories:
            temp = data[data[COLUMN_CATEGORY] == category]
            if plot_bar:
                for num_col in num_cols:
                    metric = num_col.split('_')[0].upper() + '(' + num_col.split('_')[-1].lower() + ')'
                    title = '{}: {} comparison for {} category of experiments'.format(appliance, metric,
                                                                                      category.lower())
                    to_plot = temp[cat_cols + [num_col]]
                    to_plot.loc[:, COLUMN_EXPERIMENT] = to_plot[COLUMN_EXPERIMENT].apply(experiment_name_format)
                    fig = px.bar(to_plot, x=COLUMN_EXPERIMENT, y=num_col,
                                 color=COLUMN_MODEL, barmode='group',
                                 title=title, height=400)
                    save_name = plots_save_path + PLOT_BAR + ' ' + title + PNG_EXTENSION
                    fig.write_image(save_name)

            if plot_spider:
                for experiment in temp[COLUMN_EXPERIMENT].unique():
                    title = '{}: Comparison for experiment {} ({})'.format(appliance, experiment_name_format(experiment),
                                                                           category.lower())
                    save_name = plots_save_path + PLOT_SPIDER + ' ' + title + PNG_EXTENSION
                    plot_radar_chart(data=temp[temp[COLUMN_EXPERIMENT] == experiment], num_columns=num_cols,
                                     plot_title=title, save_name=save_name)


def plot_results_from_report(appliances: list = None, report_file_path: str = None, metrics: list = None,
                             measures: list = None, categories: list = None, experiments: list = None,
                             plots_save_path: str = None):

    if measures is None:
        measures = [StatMeasures.MEAN]
    if metrics is None:
        metrics = [COLUMN_F1, COLUMN_MAE, COLUMN_RETE, COLUMN_RECALL, COLUMN_PRECISION, COLUMN_ACCURACY]
    if appliances:
        appliances = [app.upper() for app in appliances]
    if report_file_path:
        xl = pd.ExcelFile(report_file_path, engine=OPENPYXL)
        sheets = xl.sheet_names
        sheets = list_intersection(sheets, appliances)
        for sheet in sheets:
            print('#' * 40)
            print(sheet)
            print('#' * 40)
            data = xl.parse(sheet)
            plot_dataframe(data, metrics, measures, appliances, categories, experiments, plots_save_path=plots_save_path)
