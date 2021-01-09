from dateutil import parser
from collections import OrderedDict
import numpy as np
import pandas as pd

from univariate.common import tools
from univariate.common.tools import DataType, DataTypeMin
from univariate.common.woe import WoE


class UnivariateAnalysisOneVariable(object):
    """
    This class is designed for one variable univariate analysis
    """

    def __init__(self, x, y=None, dtype=None, **kwargs):
        """
        :param x: array or list like object
        :param y: the label for each element of x, optional, but is required for IV,AR calculation
        :param kwargs: these arguments will be past to WOE object, see detail in common.woe.WoE
        :param dtype: Datatype should be from DataTypeMin(enum)
        """
        # original data
        self.x = x
        # cleaned data which is of the same type
        self.clean_x = None
        # the label for each element of the data
        self.y = y
        self.woe = WoE(**kwargs)
        # ensure x and y have the same length
        if y is not None:
            assert len(y) == len(x)
            self.woe.fit(self.x, y)

        # the user defined data type
        self.dtype = dtype

        if self.dtype is not None:
            # if user provides the data type
            if not isinstance(dtype, DataTypeMin):
                # it should be from the class DataTypeMin
                raise ValueError('dtype should be an instance of DataTypeMin')

            if tools.validate_data_type(self.x, dtype):
                # ensure that the every element matches the user defined data type
                pass
            else:
                # otherwise raise error
                raise ValueError('not all input match the given dtype {}'.format(dtype))

        # get the data type count table and value container
        self._update_data_type_and_value()

    def _update_data_type_and_value(self):
        self.count, self.value = tools.array_data_type(self.x, count=True, value=True)

    @property
    def num(self):
        # number of records
        return len(self.x)

    @property
    def valid_percentage(self):
        """
        Calculate the valid number of records
        :return: float number, percentage
        """

        invalid_num = 0.0
        total_num = 0.0

        for d_type in DataType:
            # if d_type is Empty, NaN, EmptyString, or Inf, this record is considered as invalid
            if d_type in (
                    DataType.Empty, DataType.NaN, DataType.EmptyString, DataType.Inf
            ):
                invalid_num += self.count[d_type]

            total_num += self.count[d_type]

        # calculate percentage of valid records
        valid_pct = 1 - invalid_num / total_num
        return valid_pct

    @property
    def suggested_data_type(self):
        """
        The best possible data type either user provides or derived from the data
        """
        if self.dtype is not None:
            # if user provides the data type, return the data type
            return self.dtype

        temp_count = pd.Series(self.count)
        # ignore other data types with no records
        temp_count = temp_count[temp_count > 0]
        # sort the counting table, in decreasing order of the number of records.
        data_type_frequency = temp_count.sort_values(ascending=False).index

        # for each data type
        for d_type in data_type_frequency:
            if d_type in (
                DataType.Integer, DataType.Float, DataType.IntegerStoredAsString,
                DataType.FloatStoredAsString, DataType.Inf
            ):
                # if it belongs to a 'Number' group return DataTypeMin.Number
                return DataTypeMin.Number

            elif d_type == DataType.String:
                # if it's string data type, return DataTypeMin.String

                return DataTypeMin.String
            elif d_type in (
                    DataType.DateString, DataType.DateComplicateString, DataType.DateObj
            ):
                # if it's date data type, return DataTypeMin.Date
                return DataTypeMin.Date
            else:
                # other data type is not informative, such as None or empty
                # move on to the next meaningful data type
                continue
        # if None of the above, return other
        return DataTypeMin.Other

    def describe(self, strict=True):
        """
        Get the description of the data. Data will be cast first. Set strict to be True if you want to ensure all 
        the data can be cast to be of the same type
        :param strict: bool, default True. Whether all element of data needs or can be cast to be of the same type 
        :return: depends on the data type, Number, String or Date.
        """
        if self.suggested_data_type in (DataTypeMin.Number, DataTypeMin.String, DataTypeMin.Date):
            clean_x = tools.cast_data_type(self.x, self.suggested_data_type, strict=strict)
            return pd.Series(clean_x).describe()
        else:
            raise ValueError('x is not of the same supported data type')

    def cast(self, inplace=True, strict=True):
        """
        Cast the data from original data type to the proposed data type
        :param strict: bool, default True. Whether all element of data needs or can be cast to be of the same type 
        :return: 
        """
        if self.clean_x is None:
            self.clean_x = tools.cast_data_type(self.x, self.suggested_data_type, strict=strict)
            if inplace:
                self.x = self.clean_x
                self._update_data_type_and_value()
            else:
                return self.clean_x
        else:
            if not inplace:
                return self.clean_x


    @property
    def iv(self):
        return self.woe.iv

    def plot(self, *args, **kwargs):
        return self.woe.plot(*args, **kwargs)


class UnivariateAnalysis(object):
    """
    Univariate analysis for records with multiple features
    """

    def __init__(self, data, y=None, dtypes=None):
        """
        :param data: pandas Dataframe is required 
        :param y: the label for each record, optional, but is required for IV,AR calculation
        :param dtypes: a list of possible data type for each column, including 'Number', 'String', 'Date', 'Other' 
        """
        # ensure the input data is pandas DataFrame
        assert isinstance(data, pd.DataFrame)
        self.data = data
        # cleaned data which is of the same type for each column
        self.clean_data = None
        # the label for each record
        self.y = y
        # ensure x and y have the same length
        if y is not None:
            assert len(y) == data.shape[0]
        # container for one variable univariate analysis object
        self.ua_container = {}
        if dtypes is None:
            # if user doesn't provide the data type for each columns
            for col in self.data:
                # print(col)
                self.ua_container[col] = UnivariateAnalysisOneVariable(self.data[col])
        else:
            # easure that each column has a data type if provided
            assert data.shape[1] == len(dtypes)
            new_dtypes = []
            for dtype in dtypes:
                if dtype == 'Number':
                    new_dtypes.append(DataTypeMin.Number)
                elif dtype == 'String':
                    new_dtypes.append(DataTypeMin.String)
                elif dtype == 'Date':
                    new_dtypes.append(DataTypeMin.Date)
                else:
                    new_dtypes.append(DataTypeMin.Other)
            # create one variable univariate analysis object for each column
            for col, dtype in zip(self.data, new_dtypes):
                self.ua_container[col] = UnivariateAnalysisOneVariable(self.data[col], self.y, dtype)

    def basic_info(self, pct=False, row_total=True, other_info=True):
        """
        return the basic information for the data frame
        :param pct: whether return percentage or count
        :param row_total: whether show row total
        :param other_info: whether show other information, like suggested data type and valid percentage
        :return: pandas DataFrame
        """
        res_dict = OrderedDict()
        suggested = OrderedDict()
        valid_pct = OrderedDict()
        for col in self.data:
            res_dict[col] = self.ua_container[col].count
            suggested[col] = str(self.ua_container[col].suggested_data_type)
            valid_pct[col] = self.ua_container[col].valid_percentage

        res = pd.DataFrame(res_dict).T
        res.columns = [str(col) for col in res]

        total = res.sum(axis=1)

        if row_total:
            res['Total'] = total

        if pct:
            res = res.div(total, axis=0)

        if other_info:
            dtype = pd.Series(suggested)
            dtype.name = 'Suggested Data Type'
            valid_info = pd.Series(valid_pct)
            valid_info.name = 'Valid Pct'
            res = pd.concat([res, dtype, valid_info], axis=1)
        return res

    def describe(self, strict=False):
        """
        Basic description for each column
        :param strict: bool, default True. Whether all element of each column need or can be cast to be of the same type
        :return: a dictionary with Key to be Number, String, Date, and value to a pandasdataframe 
        """
        res = {}

        for col in self.data:
            dtype = self.ua_container[col].suggested_data_type

            des = self.ua_container[col].describe(strict=strict)
            des.name = col
            if str(dtype) in res:
                res[str(dtype)].append(des)
            else:
                res[str(dtype)] = [des]

        for key in res:
            res[key] = pd.concat(res[key], axis=1).T
        return res

    def cast(self, col=None, inplace=True):
        """
        Cast the data to be the same data type for each column 
        :return: 
        """
        if col is None:
            res = OrderedDict()
            if self.clean_data is None:
                for col in self.data:
                    self.ua_container[col].cast(strict=True, inplace=inplace)
                    res[col] = self.ua_container[col].clean_x
                self.clean_data = pd.DataFrame(res)
                if inplace:
                    self.data = self.clean_data
                else:
                    return self.clean_data
            else:
                if not inplace:
                    return self.clean_data
        else:
            if self.clean_data is None:
                self.ua_container[col].cast(strict=True, inplace=inplace)

                if inplace:
                    self.data[col] = self.ua_container[col].clean_x
                else:
                    return self.ua_container[col].clean_x
            else:
                if not inplace:
                    return self.clean_data[col]








if __name__ == '__main__':
    # data = pd.read_csv('data/test_data.csv', encoding='gbk')
    # x = data['本期收回'].values
    #
    # ua_one = UnivariateAnalysisOneVariable(x)
    #
    import datetime as dt
    # a = pd.Series([
    #     '20170101', '20120303', '2015-4-5', '20170101',
    #     dt.date(2017,10,2), dt.datetime(2012,3,3)
    # ])
    # ua_one = UnivariateAnalysisOneVariable(a)
    # ua_one.describe()


    b = ['20170101', '20120303', '2015-4-5']
    # tools.valide_data_type(b, DataTypeMin.Other)

    # tools.cast_data_type(b, DataTypeMin.Date)

    tools.cast_data_type([1,2,3,3,'3'], DataTypeMin.Number)

    date_b = tools.cast_data_type(b, DataTypeMin.Date)
    ua = UnivariateAnalysisOneVariable(b, dtype=DataTypeMin.Date)
    ua.describe()

    d = [1,2,3,3,'3']
    ua = UnivariateAnalysisOneVariable(b, dtype=DataTypeMin.Date)

    ua = UnivariateAnalysisOneVariable([1,2,3,'ad'])
    ua.describe(strict=False)

    data = pd.read_csv('univariate/data/sample_data.csv', encoding='gbk')
    ua = UnivariateAnalysisOneVariable(data['Order Date'])
    ua.describe(strict=False)

    print("DataFrame Input Analysis")
    ua = UnivariateAnalysis(data=data)
    res = ua.describe()

    a = ua.basic_info()

    print("单变量分析")
    import numpy as np
    x, y = np.arange(100), np.random.randint(0, 2, 100)
    ua = UnivariateAnalysisOneVariable(x, y, qnt_num=5)
    # ua.woe = WoE(qnt_num=5)

    # ua.woe.fit(x, y)

    ua.woe.plot()

    data = pd.read_csv('univariate/data/list_data.csv', encoding='gbk', index_col=0)
    print("执行")
    ua = UnivariateAnalysis(data=data)

    data = pd.read_csv('../univariate/data/sample_data.csv', encoding='gbk')

    data['label'] = [str(v) for v in np.random.randint(0, 2, len(data))]

    ua = UnivariateAnalysis(data=data)
    ua.basic_info()
    des = ua.describe()
    ua.cast(col='label', inplace=True)
    ua.basic_info()
    data = ua.data
    data['Quantity'] = tools.fill(data['Quantity'], method='median')
    ua_Quantity = UnivariateAnalysisOneVariable(x=data['Quantity'], y=data['label'], qnt_num=5, min_block_size=3)
    pd.Series(ua_Quantity.count)
    ua_Quantity.iv
    ua_Quantity.plot()
    plt.show()