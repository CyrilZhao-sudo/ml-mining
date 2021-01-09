import logging
import re
from dateutil import parser
import datetime as dt

from enum import Enum
import numpy as np
import pandas as pd


class DataType(Enum):
    """
    Enum type, support
    - Integer
    - Float
    - String
    - Empty
    - NaN
    - EmptyString
    - Inf
    - IntegerStoredAsString
    - FloatStoredAsString
    - DateString
    - DateComplicateString
    - DateObj
    - Unknown
    """

    Integer = 1
    Float = 2
    String = 3
    Empty = 4
    NaN = 5
    EmptyString = 6
    Inf = 7
    IntegerStoredAsString = 8
    FloatStoredAsString = 9
    DateString = 10
    DateComplicateString = 11
    DateObj = 12
    Unknown = 13

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value


class DataTypeMin(Enum):
    Number = 1
    String = 2
    Date = 3
    Other = 4

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value


def data_type(x):
    """
    Get the data type of one element. Supported data types can be found in DataType Enum
    :param x:  
    :return: DataType Enum
    """

    if isinstance(x, (int, np.integer)):
        # integer
        return DataType.Integer

    elif isinstance(x, (float, np.float)):
        # float
        # np.NaN and np.Inf are both np.float type
        if x.is_integer():
            # integer stored as float
            return DataType.Integer
        elif np.isnan(x):
            # np.NaN
            return DataType.NaN
        elif np.isinf(x):
            # np.Inf
            return DataType.Inf
        else:
            # real float number
            return DataType.Float

    elif isinstance(x, (str, np.str)):
        # string
        cleaned_x = clean_format(x)
        if len(x) == 0:
            # empty string ''
            return DataType.EmptyString

        elif cleaned_x.isdigit():
            # integer stored as string, '123'
            if len(cleaned_x) == 8:
                # deal with date, e.g. '20170223' stored as string
                try:
                    parser.parse(cleaned_x)
                    return DataType.DateString
                except Exception:
                    return DataType.IntegerStoredAsString

            return DataType.IntegerStoredAsString
        else:
            try:
                # check whether it can be cast to float
                # success means it's a float number stored as string
                float(cleaned_x)
                return DataType.FloatStoredAsString
            except ValueError:

                cleaned_date_x = clean_date(x)

                try:
                    parser.parse(cleaned_date_x)
                    return DataType.DateComplicateString
                except Exception:
                    # failure means it's not the case that a float number is stored as string or a date string
                    return DataType.String
    elif isinstance(x, (dt.datetime, dt.date)):
        return DataType.DateObj

    elif x is None:
        # it's None
        return DataType.Empty

    else:
        # other unsupported data type, such as object
        return DataType.Unknown


def array_data_type(x, count=True, value=False):
    """
    Count the data type in a list like object
    :param x: array or list likt object: 
    :param count: bool, default True, whether to return data type count  
    :param value: bool, default False, whether to return full list of data for each data type
    :return: dictionary or dictionary with values to be a list of tuples, which depends on 
    input parameter 
    """
    if not (count or value):
        raise ValueError('Either count or value needs to be True')

    if count:
        result_count = {
            DataType.Integer: 0,
            DataType.Float: 0,
            DataType.String: 0,
            DataType.Empty: 0,
            DataType.NaN: 0,
            DataType.Inf: 0,
            DataType.EmptyString: 0,
            DataType.IntegerStoredAsString: 0,
            DataType.FloatStoredAsString: 0,
            DataType.DateString: 0,
            DataType.DateComplicateString: 0,
            DataType.DateObj: 0,
            DataType.Unknown: 0
        }
    if value:
        result_value = {
            DataType.Integer: [],
            DataType.Float: [],
            DataType.String: [],
            DataType.Empty: [],
            DataType.NaN: [],
            DataType.Inf: [],
            DataType.EmptyString: [],
            DataType.IntegerStoredAsString: [],
            DataType.FloatStoredAsString: [],
            DataType.DateString: [],
            DataType.DateComplicateString: [],
            DataType.DateObj: [],
            DataType.Unknown: []
        }
    for i, v in enumerate(x):
        dtype = data_type(v)
        if count:
            result_count[dtype] += 1
        if value:
            result_value[dtype].append((i, v))
    if count and value:
        return result_count, result_value
    elif count:
        return result_count
    elif value:
        return result_value


def missing_value(x):
    """
    Calculate the percentage of missing value. Missing value includes 
        - DataType.EmptyString, DataType.Empty, DataType.NaN
    :param x: array or list like object
    :return: float
    """
    value_count = array_data_type(x)
    total_num = len(x)
    if total_num == 0:
        raise ValueError('length of input must be greater than 0')

    # total number of missing value
    missing = value_count[DataType.EmptyString] + \
              value_count[DataType.Empty] + \
              value_count[DataType.NaN]

    return float(missing) / total_num


def _value_count(count, value, container, value_type='numeric', cast=True):
    """
    Only count select Number or String data type
    :param count: array_data_type returned object
    :param value: array_data_type returned object
    :param container: container used to store result
    :param value_type: 
        - 'numeric' only include numerical data type 
        - 'string', only include string
        - other string, include both numeric and string data type
    :param cast: bool, default True, whether to cast string to float or integer if possible
    :return: None
    """

    def get_value(x):
        return map(lambda x: x[1], x)

    if value_type == 'numeric':
        # only select numeric value data type
        for k, v in count.items():
            if k in (DataType.Integer, DataType.Float) and v > 0:
                # integer and float
                container.extend(get_value(value[k]))

            elif cast and k == DataType.IntegerStoredAsString and v > 0:
                # if cast is turned on, string is cast to integer if possible
                container.extend(map(lambda x: int(clean_format(x)), get_value(value[k])))

            elif cast and k == DataType.FloatStoredAsString and v > 0:
                # if cast is turned on, string is cast to float if possible
                container.extend(map(lambda x: float(clean_format(x)), get_value(value[k])))

    elif value_type == 'string':
        # only select string data type
        for k, v in count.items():
            if k in (DataType.String, DataType.DateString, DataType.DateComplicateString) and v > 0:
                # string data type
                container.extend(get_value(value[k]))

            elif not cast and k == DataType.FloatStoredAsString and v > 0:
                # if cast is turned off, treat it as string even if it's float stored as string
                container.extend(get_value(value[k]))
            elif not cast and k == DataType.IntegerStoredAsString and v > 0:
                # if cast is turned off, treat it as string even if it's integer stored as string
                container.extend(get_value(value[k]))
    else:
        # include all supported data type
        for k, v in count.items():
            if k in (DataType.Integer, DataType.Float) and v > 0:
                # integer and float
                container.extend(get_value(value[k]))
            elif cast and k == DataType.IntegerStoredAsString and v > 0:
                # if cast is turned on, IntegerStoredAsString data type is cast to integer
                container.extend(map(lambda x: int(clean_format(x)), get_value(value[k])))
            elif cast and k == DataType.FloatStoredAsString and v > 0:
                # if cast is turned on, FloatStoredAsString data type is cast to integer
                container.extend(map(lambda x: float(clean_format(x)), get_value(value[k])))
            elif k in (DataType.String, DataType.DateString, DataType.DateComplicateString) and v > 0:
                # string
                container.extend(get_value(value[k]))
            elif not cast and k == DataType.FloatStoredAsString and v > 0:
                # if not cast, FloatStoredAsString is treated as string
                container.extend(get_value(value[k]))
            elif not cast and k == DataType.IntegerStoredAsString and v > 0:
                # if not cast, IntegerStoredAsString is treated as string
                container.extend(get_value(value[k]))


def describe_numeric(x, ddof=0, cast=True):
    """
    Get the count, mean, std, min, 25% quantile, 50% quantile, 75% quantile and max of a list of numeric values 
    in a array or list like object
    :param x: array or list like object 
    :param ddof: degree of freedom used in standard variance calculation, default 0
    :param cast: whether cast FloatStoredAsString or IntegerStoredAsString as a float or integer
    :return: pandas.Series
    """
    count, value = array_data_type(x, count=True, value=True)
    data = []
    # only counts numeric values
    _value_count(count, value, data, value_type='numeric', cast=cast)
    if len(data) == 0:
        return None
    data_count = len(data)
    data_unique = len(set(data))
    data_mean = np.mean(data)
    data_std = np.std(data, ddof=ddof)
    data_min = np.min(data)
    data_25p = np.percentile(data, 25)
    data_50p = np.percentile(data, 50)
    data_75p = np.percentile(data, 75)
    data_max = np.max(data)

    return pd.Series([
        data_count,
        data_unique,
        data_mean,
        data_std,
        data_min,
        data_25p,
        data_50p,
        data_75p,
        data_max,
    ], index=['count', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype='float64')


def describe(x):
    """
    Get the data type distribution for the list like object
    :param x: array and list like object 
    :return: pandas.DataFrame
    """
    count = array_data_type(x)
    total_num = len(x)
    res = pd.DataFrame.from_dict(count, orient='index', dtype=np.int64)
    res.columns = ['count']
    res['pct'] = res['count'].apply(lambda x: float(x) / total_num)
    res.index = [str(x) for x in res.index]
    return res


def distribution(x, *args, **kwargs):
    """
    Get the distribution for the numeric values
    :param x: array or list like object
    :param args: any positional arguments
    :param kwargs: any key words arguments
    :return: counts and cuts, stored as numpy.array, same as the return of numpy.histogram
    """
    count, value = array_data_type(x, count=True, value=True)
    data = []
    # only count numeric values
    _value_count(count, value, data, value_type='numeric', cast=True)
    if len(data) == 0:
        return
    else:
        return np.histogram(data, *args, **kwargs)


def unique(x, value_type='all', cast=True):
    """
    calculate unique values which only include number and string
    :param x: array or list like object
    :param value_type: 'numeric', 'string', 'all', default 'all'
    :param cast: whether cast FloatStoredAsString or IntegerStoredAsString asa float or integer
    :return: set of unique values
    """
    count, value = array_data_type(x, count=True, value=True)
    data = []
    _value_count(count, value, data, value_type, cast)
    return set(data)


def fill(x, method='median', dtype=None, value=None, cast=True, inplace=False):
    """
    Fill missing data 
    :param x: array or list like object
    :param method: default 'median', other possible option 'mean' or a function
    :param dtype: None(0), 'empty'(1), or 'empty_and_inf'(2) or DataType tuple
    :param value: values to be filled in and value has higher priority than method
    :param cast: whether to cast string to integer or float if possible
    :param inplace: whether replace value inplace or not
    :return: if not inplace, a array is returned
    """

    res = list(x)
    if dtype is None or dtype == 0:
        dtype = (
            DataType.Empty, DataType.EmptyString, DataType.Inf, DataType.NaN, DataType.Unknown
        )
    elif dtype is 'empty' or dtype == 1:
        dtype = (
            DataType.Empty, DataType.EmptyString, DataType.NaN
        )
    elif dtype is 'empty_and_inf' or dtype == 2:
        dtype = (
            DataType.Empty, DataType.EmptyString, DataType.NaN, DataType.Inf
        )
    else:
        assert isinstance(dtype, tuple)
        for v in dtype:
            assert isinstance(v, DataType)

    numeric_value = []
    none_index = []
    for i, v in enumerate(x):
        d_type = data_type(v)
        if value is None and d_type in (DataType.Float, DataType.Integer):
            numeric_value.append(v)
        elif value is None and cast and d_type in (DataType.IntegerStoredAsString, DataType.FloatStoredAsString):
            numeric_value.append(float(v))
        elif d_type in dtype:
            none_index.append(i)

    if value is None and method == 'median':
        value = np.median(numeric_value)
    elif value is None and method == 'mean':
        value = np.mean(numeric_value)
    elif value is None and hasattr(method, '__call__'):
        value = method(numeric_value)
    elif value is not None:
        d_type = data_type(value)
        if d_type in (DataType.Integer, DataType.Float):
            pass
        elif d_type in (DataType.FloatStoredAsString,):
            value = float(value)
        elif d_type in (DataType.IntegerStoredAsString,):
            value = int(value)
        else:
            raise ValueError('value is not a supported data type ({})'.format(value))
    else:
        raise ValueError('either supported method or value should be provided')

    for i in none_index:
        res[i] = value

    if inplace:
        if isinstance(x, (pd.Series, np.ndarray, list)):
            x[:] = res
        else:
            try:
                x[:] = res
            except Exception:
                logging.warning('{} is not supported for inplace replacing in our code'.format(type(x)))
                return res
    else:
        return res


def delete_or_fill_by_quantile(x, min_q=5, max_q=95, delete=False):
    """
    Keep the data only in the predefined quantiles
    :param x: array or list like object
    :param min_q: minimum quantile, default 5
    :param max_q: maximum quantile, default 95
    :return: array or list like object in the original data type
    """
    q_min = np.percentile(x, q=min_q)
    q_max = np.percentile(x, q=max_q)
    filtered = []
    index = []
    for i, v in enumerate(x):
        if q_min > v:
            if delete:
                continue
            else:
                filtered.append(q_min)
                index.append(i)
        elif q_max < v:
            if delete:
                continue
            else:
                filtered.append(q_max)
                index.append(i)
        else:
            filtered.append(v)
            index.append(i)
    if isinstance(x, pd.Series):
        return pd.Series(filtered, index=x.index[index])
    elif isinstance(x, np.ndarray):
        return np.array(filtered)
    else:
        try:
            return type(x)(filtered)
        except Exception:
            logging.warning("it's not in the original data type")
            return filtered


def delete_or_fill_by_range(x, min_x=-np.Inf, max_x=np.Inf, delete=False):
    """
    Keep the data only in the predefined range
    :param x: list or array like object
    :param min_x: lower bound, default -np.Inf
    :param max_x: upper bound, default np.Inf
    :return: array or list like object in the original data type
    """
    filtered = []
    index = []
    for i, v in enumerate(x):
        if min_x > v:
            if delete:
                continue
            else:
                filtered.append(min_x)
                index.append(i)
        elif max_x < v:
            if delete:
                continue
            else:
                filtered.append(max_x)
                index.append(i)
        else:
            filtered.append(v)
            index.append(i)
    if isinstance(x, pd.Series):
        return pd.Series(filtered, index=x.index[index])
    elif isinstance(x, np.ndarray):
        return np.array(filtered)
    else:
        try:
            return type(x)(filtered)
        except Exception:
            logging.warning("it's not in the original data type")
            return filtered


def delete_value(x, value):
    """
    Delete a or some specific values
    :param x: 
    :param value: a number or a list of numbers
    :return: data without specific values
    """
    if hasattr(value, '__iter__'):
        d_value = value
    else:
        d_value = (value,)
    filtered = []
    index = []
    for i, v in enumerate(x):
        if v not in d_value:
            filtered.append(v)
            index.append(i)
    if isinstance(x, pd.Series):
        return pd.Series(filtered, index=x.index[index])
    elif isinstance(x, np.ndarray):
        return np.array(filtered)
    else:
        try:
            return type(x)(filtered)
        except Exception:
            logging.warning("it's not in the original data type")
            return filtered


def string2number(x):
    res = []
    for v in x:
        d_type = data_type(v)
        if d_type == DataType.IntegerStoredAsString:
            res.append(int(v))
        elif d_type == DataType.FloatStoredAsString:
            res.append(float(v))
        else:
            res.append(v)
    return res


def clean_format(x):
    return re.sub(r',|%|\$', '', x)


def clean_date(x):
    return re.sub(r'年|月|日', '-', x)


def convert_data(x, convert_inf=True, parse_date=True):
    d_type = data_type(x)
    if d_type == DataType.IntegerStoredAsString:
        return int(clean_format(x))
    elif d_type == DataType.FloatStoredAsString:
        return float(clean_format(x))
    elif parse_date and d_type == DataType.DateString:
        return parser.parse(clean_date(x))
    elif d_type == DataType.EmptyString:
        return None
    elif convert_inf and d_type == DataType.Inf:
        return np.NaN
    else:
        return x


def validate_data_type(arr_obj, dtype):
    count = array_data_type(arr_obj, count=True)
    if dtype == DataTypeMin.Number:
        if count[DataType.String] + count[DataType.NaN] + count[DataType.DateComplicateString] \
                + count[DataType.DateObj] + count[DataType.Unknown] > 0:
            return False
        else:
            return True
    elif dtype == DataTypeMin.String:
        if count[DataType.Unknown] + count[DataType.NaN] + count[DataType.DateObj] > 0:
            return False
        else:
            return True
    elif dtype == DataTypeMin.Date:
        v = 0
        for k in DataType:
            if k not in (DataType.DateObj, DataType.DateString, DataType.DateComplicateString):
                v += count[k]
        return False if v > 0 else True

    raise TypeError('Not Supported Data Type')


def cast_data_type(arr_obj, dtype, strict=True):
    value = array_data_type(arr_obj, count=False, value=True)
    if strict:
        assert validate_data_type(arr_obj, dtype)
    res = []
    if dtype == DataTypeMin.Number:
        res = value[DataType.Integer] + value[DataType.Float] + \
              list(map(lambda x: (x[0], int(x[1])), value[DataType.IntegerStoredAsString])) + \
              list(map(lambda x: (x[0], float(x[1])), value[DataType.FloatStoredAsString])) + \
              list(map(lambda x: (x[0], int(x[1])), value[DataType.DateString])) + \
              list(map(lambda x: (x[0], 0), value[DataType.EmptyString] + value[DataType.Empty]))
        return list(map(lambda x: x[1], sorted(res, key=lambda x: x[0])))
    elif dtype == DataTypeMin.String:
        res = value[DataType.IntegerStoredAsString] + value[DataType.FloatStoredAsString] + \
              value[DataType.EmptyString] + value[DataType.DateString] + \
              value[DataType.String] + \
              list(map(lambda x: (x[0], ''), value[DataType.Empty])) + \
              list(map(lambda x: (x[0], str(x[1])), value[DataType.Integer] + value[DataType.Float]))
        return list(map(lambda x: x[1], sorted(res, key=lambda x: x[0])))
    elif dtype == DataTypeMin.Date:
        res = value[DataType.DateObj] + \
              list(map(lambda x: (x[0], parser.parse(clean_date(x[1]))),
                       value[DataType.DateString] + value[DataType.DateComplicateString]))
        return list(map(lambda x: x[1], sorted(res, key=lambda x: x[0])))

    raise ValueError('Not supported casting')


if __name__ == '__main__':
    array_data_type([1, 2, 3, 4, 45, '23243', '323.12', None, np.Inf, '2017-10-20', '2017年10月1日'])
    s = array_data_type([1, 2, 3, 4, 45, '23243', '323.12', None])

    describe([1, 2, 3, 4, 45, None, '2344', '', '', '2017-10-20'])

    a = np.array([1, 2, 4, 5, 6, 6, 7, 7, 7, 7, 7, 5, 5, 5, 5, None, np.Inf])
    a = pd.Series([5, 3, 4, 5, 0, None, np.Inf], index=[1, 4, 5, 7, 10, 16, 19])
    unique(a)

    fill(a, method=np.max, dtype=(DataType.Inf,))

    c = np.array([1, 2, 4, 5, 6, 6, 7, 7, 10, 20, 201, 7, 7, 5, 5, 5, 5])
    delete_or_fill_by_quantile(c)
    delete_or_fill_by_quantile([1, 2, 4, 5, 6, 6, 7, 7, 10, 20, 201, 7, 7, 5, 5, 5, 5])
    delete_or_fill_by_range([1, 2, 4, 5, 6, 6, 7, 7, 10, 20, 201, 7, 7, 5, 5, 5, 5], min_x=5, max_x=10)

    data = pd.read_csv('univariate/data/data.csv', encoding='gbk')
    data_type('bbd')
    array_data_type(data.year, count=True, value=False)
