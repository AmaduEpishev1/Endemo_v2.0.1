"""
This module contains all Enums of endemo.
"""

from __future__ import annotations

from enum import Enum, auto

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
    MULT = auto()

class DemandDriver(Enum):
    """
    The enum for dynamic input of Demand Drivers in ECU predictions (NewInstance Filter)
    """
    TIME = auto()
    POP = auto()
    GDP = auto()
    # other Ddr are can be introduced




