"""
This module contains all classes directly representing in data series and prognosis calculations.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import warnings
from typing import Any, Union
import statistics as st
import numpy as np

from endemo2.data_structures.containers import Interval, Datapoint
from endemo2.data_structures.enumerations import ForecastMethod
from endemo2 import utility as uty


class Method:
    def __init__(self):
        self.name = None  # Method used for forecasting
        self.equation = None  # Holds the regression equation
        self.coefficients: List[float] = []  # Coefficients first element is k0
        self.type_identifier = None  # Identifier for the specific row
        self.interp_points = None # A list of known data points (tuples), where each entry is a tuple of coordinates and the function value at that point.
        self.demand_drivers_names = None # list of DDr names used in the method
        self.factor = None # multiplicator to get the standard units
        self.lower_limit = None # Lower boundary for the processing variable

    def save_coef(self, coefficients: List[float], equation: str, type_identifier: str = None):
        """
        Save coefficients and the regression equation to the Method object.

        Args:
            coefficients (List[float]): Coefficients for the regression.
            equation (str): Regression equation.
            row_identifier (str, optional): Identifier for the row. Defaults to None.
        """
        self.coefficients = coefficients
        self.equation = equation
        # Optionally track the row identifier
        if type_identifier is not None:
            self.type_identifier = type_identifier

    def __str__(self):
        return (
            f"row identifier: {self.type_identifier}\n"
        )


class RigidTimeseries:
    """
    Class representing data that should be taken just like it is. Without interpolation, without coefficients,
    only the data over time.

    :param data: The timeseries data, where the x-axis is Time and the y-axis is some value over time.
        Can contain NaN and Inf values.

    :ivar [Datapoint] _data: The timeseries data, where the x-axis is Time and the y-axis is some value over time.
        Filtered to not contain any NaN or Inf values.
    """

    def __init__(self, data: [Datapoint]):
        # clean data before saving; also copies the input_and_settings data
        self._data = uty.filter_out_nan_and_inf(data)

    def __str__(self):
        return str(self._data)

    def get_last_available_data_entry_or_zero(self) -> Datapoint:
        """ Returns the last data entry in the timeseries data if present, else 0.0. """
        if len(self._data) == 0:
            return Datapoint(0.0)
        else:
            return self._data[-1]

    def get_value_at_year(self, year: int) -> float:
        """
        Returns the value of the RigidTimeseries at the given year, if present. If the value is not present, throws an
        error.

        :param year: The year the value should be taken from.
        :return: The value of the timeseries at year.
        """
        res = [y for (x, y) in self._data if x == year]

        if len(res) == 0:
            raise ValueError("Trying to access value in RigidTimeseries at a year, where no value is present.")
        else:
            return res[0]


class TwoDseries:
    """
    A class representing a data series [(x, y)]. Both axis can be any type of data.

    :param [Datapoint] data: The TwoDseries data, where the x-axis and the y-axis can be any floating point values.
        Can contain NaN and Inf values.

    :ivar [Datapoint] _data: Where the x-axis is first in tuple and the y-axis is second.
    :ivar Coef coefficients: The coefficients for this data series.
    """

    def __init__(self, data: [Datapoint]):
        # clean data before saving; also copies the input_and_settings data
        self._data = uty.filter_out_nan_and_inf(data)
        self.coefficients = None

    def __str__(self):
        return str(self._data)

    def generate_coef(self) -> Method:
        """
        Generate and save the coefficients of all regression methods.

        :return: The generated Coefficient object.
        """
        self.coefficients = uty.apply_all_regressions(self._data)

        if len(self._data) == 1:
            self.coefficients.set_exp_start_point(self._data[0])
            self.coefficients.set_method(ForecastMethod.EXP, fixate=True)

        return self.coefficients

    def get_coef(self) -> Method:
        """
        Safely get the coefficients. If they were not generated before, generates them.

        :return: The resulting coefficient object.
        """
        if self.coefficients is None:
            return self.generate_coef()
        else:
            return self.coefficients

    def is_empty(self) -> bool:
        """
        Indicates whether there is no data present.

        :return: True if data is empty, False otherwise.
        """
        return len(self._data) == 0

    def is_zero(self) -> bool:
        """
        Indicates whether the data on the y-axis has only zeroes.

        :return: True if data's y-axis only contains zeros, False otherwise.
        """
        return uty.is_tuple_list_zero(self._data)

    def get_data(self) -> [Datapoint]:
        """ Getter for the data attribute. """
        return self._data

    def get_mean_y(self) -> float:
        """
        Get mean of all values on the y-axis of the data.

        :return: The mean of all y values of the data.
        """
        only_y = [y for (x, y) in self._data]
        return st.mean(only_y)

    def append_others_data(self, other_tds: TwoDseries) -> TwoDseries:
        """
        Append all data of another timeseries to self and return self.

        :param other_tds: The TwoDseries, whose data should be appended.
        :return: self
        """
        self._data += other_tds._data
        return self


class Timeseries(TwoDseries, RigidTimeseries):
    """
    A class strictly representing a value over time. This usage is a class invariant and the class should not be used
    in any other way.

    :param [Datapoint] data: The timeseries data, where the x-axis is Time and the y-axis is some value over time.
        Can contain NaN and Inf values.

    :ivar [Datapoint] _data: Where the x-axis is Time and the y-axis is some value over time.
    """

    def __init__(self, data: [Datapoint]):
        super().__init__(data)

    @classmethod
    def merge_two_timeseries(cls, t1: Timeseries, t2: Timeseries) -> TwoDseries:
        """
        Merge the two Timeseries on the condition that the year is the same.
        Note: The result is not a Timeseries!

        :param t1: Timeseries to zip t1 [(a,b)]
        :param t2: Timeseries to zip t2 [(c, d)]
        :return: The zipped TwoDseries in format [(b, d)] for where a == c.
        """
        d1 = t1._data
        d2 = t2._data
        return TwoDseries(uty.zip_data_on_x(d1, d2))

    @classmethod
    def map_two_timeseries(cls, t1: Timeseries, t2: Timeseries, function) -> Any:
        """
        Map the two Timeseries according to the given function.

        :param t1: Timeseries to zip t1 [(x, y1)]
        :param t2: Timeseries to zip t2 [(x, y2)]
        :param function: The function that is applied. Should be of form lambda x y1 y2 -> ...
        :return: The result of the applied function.
        """
        d1 = t1._data
        d2 = t2._data
        return TwoDseries(uty.zip_data_on_x_and_map(d1, d2, function))

    @classmethod
    def map_y(cls, t: Timeseries, function) -> Timeseries:
        """
        Map the y-axis of the Timeseries and return a newly created timeseries as a result.

        :param t: The timeseries to map.
        :param function: The mapping function of the form lambda x -> ...
        :return: A copy-mapped timeseries.
        """
        return Timeseries(uty.map_data_y(t._data, function))

    def append_others_data(self, other_ts: Timeseries) -> Timeseries:
        """
        Append all data of another timeseries to self and return self.
        Timeseries also sorts the result, so it can remain a valid timeseries.

        :param other_ts: The timeseries, whose data should be appended.
        :return: self
        """
        super().append_others_data(other_ts)
        self._data.sort(key=lambda data_point: data_point.x)
        return self

    def add(self, other_ts: Timeseries) -> Timeseries:
        """
        Add the data of the other_ts to self. Also return a reference to self.
        When a datapoint is only present in one of the timeseries it is added!

        :param other_ts: The timeseries, whose data should be added to self.
        :return: A reference to self.
        """
        (start, end) = uty.get_series_range([self, other_ts])
        other_ts = Timeseries.fill_empty_years_with_value(other_ts, Interval(start, end), np.NaN)
        self.fill_own_empty_years_with_value(Interval(start, end), np.NaN)

        zipped = list(zip(other_ts._data, self._data))
        added_data = []
        for (x, y1), (_, y2) in zipped:
            if y1 is np.NaN and y2 is np.NaN:
                continue
            y1_nan_to_zero = 0.0 if y1 is np.NaN else y1
            y2_nan_to_zero = 0.0 if y2 is np.NaN else y2
            added_data.append(Datapoint(x, y1_nan_to_zero + y2_nan_to_zero))

        self._data = added_data
        return self

    def divide_by(self, other_ts: Timeseries) -> Timeseries:
        """
        Divide the data of self by the data of other_ts. Also return a reference to self.

        :param other_ts: The timeseries, whose data should be divided by.
        :return: A reference to self.
        """
        others_data_without_zeros = [Datapoint(x, y) for (x, y) in other_ts._data if y != 0]
        self._data = uty.zip_data_on_x_and_map(self._data, others_data_without_zeros,
                                               lambda x, y1, y2: Datapoint(x, y1 / y2))
        return self

    def scale(self, scalar: float) -> Timeseries:
        """
        Scale own data by scalar. Also return a reference to self.

        :param scalar: The scalar that should be scaled by.
        :return: A reference to self.
        """
        self._data = [Datapoint(x, y * scalar) for (x, y) in self._data]
        return self

    def get_last_data_entry(self) -> Datapoint:
        """
        Getter for the last available entry in data.

        :return: The last entry, or a string if data is empty.
        """
        if self.is_empty():
            return "", ""
        else:
            return self._data[-1]

    def get_value_at_year(self, year: int) -> float:
        """
        Returns the value of the RigidTimeseries at the given year, if present. If the value is not present, throws an
        error.

        :param year: The year, the data should be taken from.
        :return: The value of the timeseries at year.
        """
        res = [y for (x, y) in self._data if x == year]

        if len(res) == 0:
            raise ValueError("Trying to access value in RigidTimeseries at a year, where no value is present.")
        else:
            return res[0]

    def get_value_at_year_else_zero(self, year: int) -> float:
        """
        Returns the value of the RigidTimeseries at the given year, if present. If the value is not present, returns
        zero.

        :param year: The year, the data should be taken from.
        :return: The value of the timeseries at year, or zero if no data is present.
        """
        res = [y for (x, y) in self._data if x == year]

        if len(res) == 0:
            return 0.0
        else:
            return res[0]

    def fill_own_empty_years_with_value(self, interval: Interval, fill_value: float):
        """
        Fill own data in interval with fill value

        :param interval: The interval that should be filled.
        :param fill_value: The value that should be used to fill gaps
        """
        result = []
        current_year = interval.start
        for year, value in self._data:
            # add zeros before the next year that is in timeseries to fill gaps
            while current_year < year:
                result.append(Datapoint(current_year, fill_value))
                current_year += 1
            result.append(Datapoint(year, value))
            current_year += 1

        while current_year <= interval.end:
            result.append(Datapoint(current_year, fill_value))
            current_year += 1

        self._data = result

    @classmethod
    def fill_empty_years_with_value(cls, ts: Timeseries, interval: Interval, fill_value: float) -> Timeseries:
        """
        Create a timeseries that has data for each year in interval. Take value from given timeseries if present, else
        fill with given value.

        :param ts: The given timeseries, which values are copied.
        :param interval: The interval in which data has to be present for every year.
        :param fill_value: Value with which gaps in the data should be filled.
        :return: The created timeseries with values from ts and filled gaps with given value.
        """
        result = Timeseries(ts._data.copy())
        result.fill_own_empty_years_with_value(interval, fill_value)

        return result


class IntervalForecast:
    """
    The class depicting a given exponential prediction with different growth rates in certain intervals.

    :param list[(Interval, float)] progression_data: The input_and_settings progression data given as a list of
        intervals and their corresponding growth rate. For example [(Interval(start, end), percentage_growth)].

    :ivar list[(Interval, float)] _interval_changeRate: The same as progression_data, just the growth rate is not in
        percentage anymore, but percentage/100
    """

    def __init__(self, progression_data: list[(Interval, float)]):
        # map percentage to its hundredth
        self._interval_changeRate = [(prog[0], prog[1] / 100) for prog in progression_data]

    def get_forecast(self, target_x: float, start_point: Datapoint) -> float:
        """
        Get the prognosis of the y-axis value for a target x-axis value from the manual exponential
        interval-growth-rate forecast.

        .. math::
            y=s_x*(1+r_{1})^{(intvl^{(1)}_{b}-s_y)}*(1+r_{2})^{(intvl^{(2)}_{b}-intvl^{(2)}_{a})}*\\text{...}*(1+r_{3})^
            {(x-intvl^{(3)}_{a})}

        :param start_point: The (x, y) Tuple, that is used as the first value for the exponential growth.
        :param target_x: The target x-axis value.
        :return: The predicted y value at x-axis value x.
        """
        result = start_point.y
        for interval_change in self._interval_changeRate:
            start = max(start_point.x, interval_change[0].start)  # cut off protruding years at start
            end = min(target_x, interval_change[0].end)  # cut off protruding years at end
            exp = max(0, end - start)  # clamp to 0, to ignore certain intervals
            result *= (1 + interval_change[1]) ** exp

        return result

