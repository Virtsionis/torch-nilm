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
    OVEN = 'oven'
    LIGHT = 'light'
    PRINTER = 'printer'
    BOILER = 'boiler'
    ELECTRIC_HEATER = 'electric space heater'
    TELEVISION = 'television'
    TOASTER = 'toaster'
    IMMERSION_HEATER = 'immersion heater'
    WATER_PUMP = 'water pump'


class SupportedExperimentVolumes(Enum):
    LARGE_VOLUME = 'large'
    COMMON_VOLUME = 'common'
    SMALL_VOLUME = 'small'
    CV_VOLUME = 'cv'
    UKDALE_ALL_VOLUME = 'ukdale_all'


class SupportedPreprocessingMethods(Enum):
    ROLLING_WINDOW = 'rolling_window'
    MIDPOINT_WINDOW = 'midpoint_window'
    SEQ_T0_SEQ = 'sequence-to-sequence'
    SEQ_T0_SUBSEQ = 'sequence-to-sub-sequence'


class SupportedFillingMethods(Enum):
    FILL_ZEROS = 'fill_zeros'
    FILL_INTERPOLATION = 'fill_interpolation'


class SupportedScalingMethods(Enum):
    STANDARDIZATION = 'standardization'
    NORMALIZATION = 'normalization'
    MIXED = 'mixed'

