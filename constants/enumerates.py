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
