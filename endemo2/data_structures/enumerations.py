"""
This module contains all Enums of endemo.
"""

from __future__ import annotations

from enum import Enum, auto


# class SubsectorGroup(Enum):
#     """
#     Enum to group the industry _subsectors.
#     """
#     CHEMICALS_AND_PETROCHEMICALS = 0
#     FOOD_AND_TOBACCO = 1
#     IRON_AND_STEEL = 2
#     NON_METALIC_MINERALS = 3
#     PAPER = 4
#
#
# class GroupType(Enum):
#     """
#     Enum to differentiate different country group types.
#
#     :ivar SEPARATE: All countries in this type of group are calculated separately.
#     :ivar JOINED: The data of all countries in a group of type joined is lumped together and shared coefficients are
#         calculated.
#     :ivar JOINED_DIVERSIFIED: The data of all countries in a group of type joined_diversified is lumped together and
#         shared coefficients are calculated. But each country has a differing offset additionally.
#     :ivar EMPTY: Indicates that no group type is chosen.
#     """
#     SEPARATE = 0
#     JOINED = 1
#     JOINED_DIVERSIFIED = 2
#     EMPTY = 3
#
# class DataOrigin(Enum):  #pereimenovat   DATAORIGIN
#     Historical = auto()
#     User = auto()

class ForecastMethod(Enum):
    LIN = auto()
    EXP = auto()
    LOG = auto()
    CONST  = auto()
    CONST_MEAN = auto()
    CONST_LAST = auto()
    QUADR = auto()
    POLY = auto()
    CONST_MULT_DIV = auto()
    LIN_MULT_DIV = auto()
    EXP_MULT_DIV = auto()
    INTERP_LIN   = auto()


# class ScForecastMethod(Enum):
#     """ The ScForecastMethod indicates type of forecast for specific consumption, primarily in the CTS sector. """
#     LINEAR = 0
#     LOGARITHMIC = 1
#     CONST_MEAN = 2
#     CONST_LAST = 3
#
#
# class DemandType(Enum):
#     """ Enum to easily differentiate the type of demand. """
#     ELECTRICITY = 0
#     HEAT = 1
#     HYDROGEN = 2
#     FUEL = 3



class DemandDriver(Enum):
    """
    The enum for dynamic input of Demand Drivers in ECU predictions (NewInstance Filter)
    """
    TIME = auto()
    POP = auto()
    GDP = auto()
    # other Ddr are can be introduced




