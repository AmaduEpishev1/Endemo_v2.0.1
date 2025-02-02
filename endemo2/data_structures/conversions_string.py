from __future__ import annotations
from endemo2.data_structures.enumerations import ForecastMethod

# Mapping ForecastPrediction Enum to strings for method selections
map_forecast_method_to_string = {
    ForecastMethod.LIN: "lin",
    ForecastMethod.EXP: "exp",
    ForecastMethod.LOG: "log",
    ForecastMethod.CONST_MEAN: "const_mean",
    ForecastMethod.CONST_LAST: "const-last",
    ForecastMethod.CONST: "const",
    ForecastMethod.QUADR: "quadr",
    ForecastMethod.POLY: "poly",
    ForecastMethod.LIN_MULT_DIV: "lin-mult-dev",
    ForecastMethod.EXP_MULT_DIV: "exp-mult-div",
    ForecastMethod.CONST_MULT_DIV: "const_mult_div",
    ForecastMethod.INTERP_LIN: "interp_lin",
    ForecastMethod.MULT: "mult"
}



