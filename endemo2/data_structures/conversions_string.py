from __future__ import annotations

from endemo2.data_structures.enumerations import ForecastMethod

# map_demand_to_string = {DemandType.ELECTRICITY: "electricity",
#                         DemandType.HEAT: "heat",
#                         DemandType.HYDROGEN: "hydrogen",
#                         DemandType.FUEL: "fuel"}
#
#
# # Mapping DemandDriver Enum to strings
# map_demand_driver_to_string = {
#     DemandDriver.POP: "POP",
#     DemandDriver.GDP: "GDP",
#     DemandDriver.TIME: "TIME"
# }

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
    ForecastMethod.LIN_MULT_DIV: "lin_mult_div",
    ForecastMethod.EXP_MULT_DIV: "exp_mult_div",
    ForecastMethod.CONST_MULT_DIV: "const_mult_div",
    ForecastMethod.INTERP_LIN: "interp_lin",
}

# # Map for forecast Data
# map_forecast_data_origin_to_string = {
#     DataOrigin.Historical: "Historical",
#     DataOrigin.User: "User"
# }
#
# #Map for Sectors


