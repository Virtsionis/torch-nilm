from enum import Enum


class DataTypes(Enum):
    INT = 'int'
    INT64 = 'int64'
    FLOAT = 'float'
    FLOAT64 = 'float64'
    OBJECT = 'object'


class StatMeasures(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    STANDARD_DEVIATION = 'std'
    MINIMUM = 'min'
    MAXIMUM = 'max'
    PERCENTILE_25TH = '25th_percentile'
    PERCENTILE_75TH = '75th_percentile'


class SupportedNilmExperiments(Enum):
    BENCHMARK = 'benchmark'
    CROSS_VALIDATION = 'cross_validation'
    HYPERPARAM_TUNE_CV = 'hyperparameter_tuning_cross_validation'


class SupportedExperimentCategories(Enum):
    SINGLE_CATEGORY = 'Single'
    MULTI_CATEGORY = 'Multi'


class ElectricalAppliances(Enum):
    MICROWAVE = 'microwave'
    KETTLE = 'kettle'
    FRIDGE = 'fridge'
    WASHING_MACHINE = 'washing machine'
    DISH_WASHER = 'dish washer'
    TUMBLE_DRYER = 'tumble dryer'
    COMPUTER = 'computer'
    OVEN = 'electric oven'
    LIGHT = 'light'
    ELECTRIC_HEATER = 'electric space heater'


class SupportedExperimentVolumes(Enum):
    LARGE_VOLUME = 'large'
    SMALL_VOLUME = 'small'
    CV_VOLUME = 'cv'


class SupportedPreprocessingMethods(Enum):
    ROLLING_WINDOW = 'rolling_window'
    MIDPOINT_WINDOW = 'midpoint_window'
    SEQ_T0_SEQ = 'sequence-to-sequence'
    SEQ_T0_SUBSEQ = 'sequence-to-sub-sequence'


class SupportedFillingMethods(Enum):
    FILL_ZEROS = 'fill_zeros'
    FILL_INTERPOLATION = 'fill_interpolation'

