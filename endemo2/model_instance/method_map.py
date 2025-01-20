from endemo2.data_structures.enumerations import ForecastMethod
from endemo2.data_structures import prediction_methods as pm
from endemo2.model_instance import calc_functions as ff, calc_coeff as cg

forecast_methods_map = {
    ForecastMethod.CONST_MULT_DIV: {
        "generate_coef": "",
        "min_points": 1,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff.calculate_const_mult_div,
        "get_eqaution_user": "y = k0 * (X1 / X2)."
    },
    ForecastMethod.LIN_MULT_DIV: {
        "generate_coef":"",
        "min_points": 1,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff.calculate_lin_mult_div,
        "get_eqaution_user": "y = (k0 + k1 * X1) * X2 / X3."
    },
    ForecastMethod.EXP_MULT_DIV: {
        "generate_coef": "",
        "min_points": 1,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff.exp_mult_div,
        "get_eqaution_user": "y = k0 + k1 * ((1 + k2 / 100) ^ (X1 - k3)) * X2 / X3"
    },
    ForecastMethod.LIN: {
        "generate_coef": cg.generate_linear_coefficients_multivariable_sklearn,
        "min_points": 2,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff._predict_lin,
        "get_eqaution_user": "y = k0 + k1*x1+k2*x2+…"
    },
    ForecastMethod.LOG: {
        "generate_coef": cg.generate_logarithmic_coefficients_multivariable,
        "min_points": 2,
        "save_coef": pm.Method.save_coef,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.QUADR: {
        "generate_coef": cg.generate_quadratic_coefficients_multivariable,
        "min_points": 3,
        "save_coef": pm.Method.save_coef,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.CONST_LAST: {
        "generate_coef": cg.generate_constant_last_coefficient_,
        "min_points": 1,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff._predict_constant_last,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.CONST_MEAN: {
        "generate_coef": cg.generate_constant_mean_coefficient_,
        "min_points": 1,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff._predict_constant_mean,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.EXP: {
        "generate_coef": cg.calculate_exponential_multivariable_coefficients,
        "min_points": 2,
        "save_coef": pm.Method.save_coef,
        "predict_function": ff.exponential_multivariable,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.POLY: {
        "generate_coef": lambda X, y: cg.generate_polynomial_coefficients_multivariable(X, y, degree=3),
        "min_points": 3,
        "save_coef": pm.Method.save_coef,
        "get_eqaution_user": "y = ...."
    },
    ForecastMethod.INTERP_LIN: {
        "generate_coef":"",
        "min_points":"",
        "save_coef": "",
        "predict_function": ff.multivariable_interpolation,
        "get_eqaution_user": "linear interpolation"
    }
}
















