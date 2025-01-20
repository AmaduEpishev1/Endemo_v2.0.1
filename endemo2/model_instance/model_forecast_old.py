import pandas as pd
import os
import logging
from endemo2.data_structures import prediction_methods as pm
import numpy as np
from endemo2.input_and_settings import input_loader as uty
from endemo2.input_and_settings.input_manager import InputManager
from endemo2.data_structures.conversions_string import  map_forecast_method_to_string
from endemo2.model_instance.method_map import  forecast_methods_map
from endemo2.data_structures.enumerations import ForecastMethod

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Forecast:

    def __init__(self, input_data, demand_driver_data,general_settings):
        """
        Initialize the GeneralInstanceFilter.
        :param input_data: The hierarchy of sectors (list of Sector objects).
        :param demand_driver_data: The DemandDriver instance holding global DDr values.
        :param coef_and_forecast: is a dictionary where the keys are sector names (IND, HOU, etc.).
        Each sector contains a dictionary where keys are subsector names (STEEL_PRIM, SP_HEAT, etc.) and values are Pandas DataFrames.
        """
        self.input_data = input_data
        self.demand_driver_data = demand_driver_data
        self.general_settings = general_settings
        self.sectors_predictions = self.process()
        self.write_sector_predictions_to_excel()

    def process(self):
        """
        Process ECU and DDet variables to prepare coefficients and generate predictions.

        :return: A dictionary with sector names as keys and DataFrames of predictions as values.
        """
        sectors_predictions_df_list = {}

        for sector in self.input_data:
            sector_name = sector.name  # Capture the sector name
            sector_predictions = []  # Initialize list to accumulate subsector predictions for this sector

            for subsector in sector.subsectors:
                subsector_predictions = []  # Collect predictions for this subsector
                subsector_name = subsector.name  # Capture the subsector name

                # Process ECU Variable
                self._forecast_variable(subsector.ecu, subsector_predictions, sector_name, subsector_name, None)

                # Process DDet Variables under Technologies
                for technology in subsector.technologies:
                    technology_name = technology.name  # Capture the technology name
                    for variable in technology.ddets:
                        self._forecast_variable(variable, subsector_predictions, sector_name, subsector_name,
                                                technology_name)

                # If we have data for the subsector, store it under the sector
                if subsector_predictions:
                    sector_predictions.append(pd.concat(subsector_predictions, ignore_index=True))

            # Combine all subsector predictions for this sector into a single DataFrame
            if sector_predictions:
                sectors_predictions_df_list[sector_name] = pd.concat(sector_predictions, ignore_index=True)

        return sectors_predictions_df_list

    def _forecast_variable(self, variable, subsector_predictions, sector_name, subsector_name, technology_name):
        """
        Process a single variable, calculate coefficients, and generate predictions.

        :param variable: Variable object (ECU or DDet).
        :param subsector_predictions: List to collect predictions for the subsector.
        :param sector_name: Name of the sector.
        :param subsector_name: Name of the subsector.
        :param technology_name: Name of the technology (None for ECU variables).
        """
        for region_data in variable.region_data:
            region_name = region_data.region_name

            if region_data.settings is None or region_data.settings.empty:
                logger.warning(f"No settings for {region_name}. Skipping variable: {variable.name}.")
                continue

            forecast_method, demand_drivers = self._extract_forecast_settings(region_data)
            if not forecast_method:
                logger.warning(f"Forecast method missing for {region_name}. Skipping variable: {variable.name}.")
                continue

            # Iterate over energy types (if any) 
            if region_data.types:
                for typ, data_dict in region_data.types.items():
                    historical_data = data_dict.get("historical")
                    user_data = data_dict.get("user")
                    historical_data = uty.clean_dataframe(historical_data)
                    user_data = uty.clean_dataframe(user_data)
                    region_data.forecast = self._forecast_variable_region(historical_data, user_data, forecast_method, demand_drivers,
                                               region_name, variable.name, subsector_predictions,
                                               sector_name, subsector_name, technology_name)
            else:
                # Handle ECU Variables directly
                region_data.forecast = self._forecast_variable_region(region_data.historical, region_data.user, forecast_method,
                                           demand_drivers, region_name, variable.name, subsector_predictions,
                                           sector_name, subsector_name, technology_name)


    def _forecast_variable_region(self, historical_data, user_data, forecast_method, demand_drivers,
                              region_name, variable_name, subsector_predictions,
                              sector_name, subsector_name, technology_name):
        """
        Generate predictions based on historical or user data.

        :param historical_data: Historical data (DataFrame or None).
        :param user_data: User forecast data (DataFrame or None).
        :param forecast_method: ForecastMethod enum.
        :param demand_drivers: List of demand drivers.
        :param region_name: Name of the region.
        :param variable_name: Name of the variable being processed.
        :param subsector_predictions: List to collect predictions for the subsector.
        :param sector_name: Name of the sector.
        :param subsector_name: Name of the subsector.
        :param technology_name: Name of the technology (None for ECU variables).
        """

        # Safely check if historical_data is a valid DataFrame
        if isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
            coefficients = self._calc_coeff_hist(
                historical_data, forecast_method, demand_drivers, region_name
            )
        elif isinstance(user_data, pd.DataFrame) and not user_data.empty:
            coefficients = self._calc_coeff_user(
                user_data, forecast_method, demand_drivers, region_name, variable_name
            )
        else:
            logger.warning(f"No valid data available for {region_name}, {variable_name}. Skipping...")
            return

        # Proceed only if coefficients are valid
        if coefficients:
            prediction = self._do_predictions(coefficients, forecast_method, demand_drivers, region_name)
            prediction_df = prediction.copy()
            # Add hierarchy metadata columns for structure
            prediction_df["Sector"] = sector_name
            prediction_df["Subsector"] = subsector_name
            prediction_df["Technology"] = technology_name if technology_name else "None"
            prediction_df["Variable"] = variable_name  # TODO normolize variable name here

            column_order = ["Region", "Sector", "Subsector","Variable", "Technology"] + [
                col for col in prediction_df.columns if col not in ["Region", "Sector", "Subsector","Variable", "Technology"]
            ]
            prediction_df = prediction_df[column_order]
            # Append predictions to subsector_predictions
            subsector_predictions.append(prediction_df)
            return prediction
            

    def _extract_forecast_settings(self, region_data):
        settings_data = region_data.settings
        forecast_method_str = (
            settings_data.iloc[0]['Function'].strip().lower()
            if 'Function' in settings_data.columns else None
        )
        forecast_method = next(
            (method for method, method_str in map_forecast_method_to_string.items()
             if method_str == forecast_method_str),
            None
        )

        demand_drivers = [
            settings_data.iloc[0][col].strip()
            for col in settings_data.columns
            if col.startswith("DDr") and not pd.isna(settings_data.iloc[0][col])
        ]

        if not forecast_method:
            logging.warning(f"Invalid or missing forecast method for {region_data.region_name}. Skipping...")
        return forecast_method, demand_drivers

    def _calc_coeff_user(self, user_data: pd.DataFrame, forecast_method: ForecastMethod,
                           demand_drivers: list, region_name: str, variable_name: str) -> pm.Method:
        """
        Process user data for a region, identifying coefficients and future projections.

        :param user_data: DataFrame containing user-specific forecast data.
        :param forecast_method: Forecasting method to use.
        :param demand_drivers: List of demand drivers.
        :param region_name: Name of the region.
        :param variable_name: Name of the variable being processed.
        :return: Coef object containing the calculated coefficients or projections.
        """
        coef = pm.Method()

        if user_data is None or user_data.empty:
            logger.warning(f"User data for {variable_name} in region {region_name} is missing or empty.")
            return pm.Method()

        # Extract relevant keys for coefficients and years
        coef_keys = [col for col in user_data.columns if isinstance(col, str) and col.startswith('k')]
        year_keys = [str(col) for col in user_data.columns if isinstance(col, str) and col.isdigit()]

        has_user_given_coef = False
        has_future_values = False
        future_data = []

        # Check for future projections
        if year_keys:
            future_data = self._extract_future_projections(user_data, year_keys)
            has_future_values = bool(future_data)

        # Check for user-provided coefficients
        if coef_keys:
            user_coefficients = self._extract_user_given_coefficients(user_data, coef_keys)
            has_user_given_coef = bool(user_coefficients)

        # Scenario 1: Only future projections provided
        if has_future_values and not has_user_given_coef:
            return self._process_future_projections(future_data, forecast_method, demand_drivers, region_name)

        # Scenario 2: Only user coefficients provided
        if has_user_given_coef and not has_future_values:
            return self.save_coef_user(user_coefficients = user_coefficients,forecast_method = forecast_method)

        # Scenario 3: Insufficient data
        if not has_future_values and not has_user_given_coef:
            logging.warning(f"Insufficient data for region: {region_name}, variable: {variable_name}. Skipping...")
            return pm.Method()

        return coef

    def save_coef_user(self, user_coefficients, forecast_method):
        method_details = forecast_methods_map[forecast_method]
        equation = method_details["get_eqaution_user"]
        save_coef = method_details["save_coef"]

        if isinstance(user_coefficients, float):
            user_coefficients = [user_coefficients]

        try:
            # Save coefficients into the Coef object
            coef_object = pm.Method()
            save_coef(coef_object,user_coefficients,equation)

            return coef_object
        except Exception as e:
            print(f"Error saving user given coefficients for method '{forecast_method}': {e}")
            return None

    def _extract_future_projections(self, user_data: pd.DataFrame, year_keys: list) -> list:
        """
        Extract future projections from the user data.

        :param user_data: DataFrame containing user data.
        :param year_keys: List of columns representing future years.
        :return: List of tuples (year, value) for future projections.
        """
        projections = []
        for year in year_keys:
            try:
                value = float(user_data.iloc[0][year])
                projections.append((int(year), value))
            except ValueError:
                logging.warning(f"Invalid value for year {year} in user data.")
        return projections

    def _calc_coeff_hist(self, historical_data, forecast_method: ForecastMethod,
                                                demand_drivers: list, region_name: str) -> pm.Method:
        """
        Calculate coefficients from historical data and demand drivers.

        :param historical_data: DataFrame containing historical values (Region + years).
        :param forecast_method: Forecast method to use (e.g., linear, quadratic).
        :param demand_drivers: List of demand driver names.
        :param region_name: Name of the region for which data is calculated.
        :return: Coefficient object.
        """
        if historical_data.empty:
            raise ValueError(f"Historical data for {region_name} is empty.")

        # Separate year columns from non-year columns
        year_columns = [col for col in historical_data.columns if str(col).isdigit()]  # Identify valid year columns
        non_year_columns = [col for col in historical_data.columns if col not in year_columns]  # Non-year columns

        if not year_columns:
            raise ValueError(f"No valid year columns found in historical data for {region_name}.")

        # Ensure year columns are integers
        year_columns = list(map(int, year_columns))

        # Set the non-year columns as index temporarily
        historical_data = historical_data.set_index(non_year_columns)

        # Validate column alignment
        # if len(historical_data.columns) != len(year_columns):
        #     raise ValueError(f"Mismatch between year columns ({len(year_columns)}) and historical data columns "
        #                      f"({len(historical_data.columns)}).")

        historical_data.columns = year_columns  # Assign year columns as integers

        # Get filtered demand driver data
        demand_driver_data = self._map_filtered_demand_driver_data(demand_drivers, year_columns, region_name,
                                                                   data_origin="historical")

        # Align historical data with demand driver years
        filtered_years = demand_driver_data.index.astype(int)  # Ensure filtered years are integers
        common_years = sorted(set(filtered_years).intersection(historical_data.columns))  # Find common years

        if not common_years:
            raise ValueError(f"No common years between demand drivers and historical data for {region_name}")

        # Filter historical data and demand driver data to only include common years
        historical_values = historical_data[common_years].iloc[0].values.tolist()  # Select region and years
        demand_driver_array = demand_driver_data.loc[common_years].values  # Align demand driver data

        # Pass the filtered data and aligned demand driver data for coefficient calculation
        return self._calculate_coef_for_filtered_data(
            values=historical_values,
            demand_driver_data=demand_driver_array,
            forecast_method=forecast_method
        )

    def _calculate_coef_for_filtered_data(self, values, demand_driver_data, forecast_method: ForecastMethod) -> pm.Method:
        """
        Calculate coefficients for the given values and demand driver data using the specified forecast method.

        :param values: List of y-values (dependent variable).
        :param demand_driver_data: 2D array of x-values (independent variables).
        :param forecast_method: The ForecastMethod enum specifying which method to use for coefficient generation.
        :return: A Coef object containing the calculated coefficients.
        """
        coef = pm.Method()  # Initialize an empty Coef object

        # Handle the case where only one value exists (constant forecast)
        if len(values) == 1:
            forecast_method = ForecastMethod.CONST_LAST
            coef.forecast_method = ForecastMethod.CONST_LAST

        # Check if the forecast method exists in the map
        if forecast_method not in forecast_methods_map:
            print(f"Warning: Forecast method '{forecast_method}' is not recognized. Using 'CONST_LAST' as fallback.")
            forecast_method = ForecastMethod.CONST_LAST

        # Retrieve the method details from the map
        method_details = forecast_methods_map[forecast_method]
        generate_coef = method_details["generate_coef"]
        min_points = method_details["min_points"]

        # Validate the number of data points
        if len(values) < min_points:
            print(
                f"Warning: Insufficient data points for method '{forecast_method.name}'. Minimum required: {min_points}."
            )
            return coef

        try:
            # Ensure demand_driver_data is a 2D array
            if len(demand_driver_data.shape) == 1:
                demand_driver_data = demand_driver_data.reshape(-1, 1)

            # Generate coefficients using the mapped function
            coefficients,equation = generate_coef(X=demand_driver_data, y=values)

            # Save coefficients into the Coef object
            coef.save_coef(coefficients,equation)

            return coef
        except Exception as e:
            print(f"Error generating coefficients for method '{forecast_method}': {e}")
            return coef

    def _map_filtered_demand_driver_data(self, demand_drivers, years, region_name, data_origin):
        aligned_DDr_df = pd.DataFrame(index=years)

        for driver_name in demand_drivers:
            if driver_name == "TIME":
                aligned_DDr_df["TIME"] = years
                continue

            driver_data_object = self.demand_driver_data.get_driver_data(driver_name)
            if not driver_data_object:
                print(f"Demand driver '{driver_name}' not found.")
                aligned_DDr_df[driver_name] = np.nan
                continue

            region_data = driver_data_object.get_data_for_region(region_name, data_origin)
            if region_data is not None and not region_data.empty:
                filtered_values = self._filter_values_by_year(region_data, years)
                aligned_DDr_df[driver_name] = filtered_values
            else:
                print(f"No data for demand driver '{driver_name}' in region '{region_name}'.")
                aligned_DDr_df[driver_name] = np.nan

        return aligned_DDr_df

    def _filter_values_by_year(self, region_data: pd.DataFrame, years: list) -> list:
        """
        Filter region data to extract values corresponding to specified years.

        :param region_data: DataFrame containing region-specific demand driver data.
        :param years: List of years to filter the data by.
        :return: List of values corresponding to the specified years.
        """
        # Ensure column names are strings to handle year columns
        region_data.columns = region_data.columns.map(str)

        # Find matching year columns
        matching_columns = [str(year) for year in years if str(year) in region_data.columns]

        if not matching_columns:
            print(f"Warning: None of the specified years '{years}' found in the data columns.")
            return []

        # Extract values for the matching years
        filtered_values = region_data.iloc[0][matching_columns].astype(float).tolist()
        return filtered_values

    def _process_future_projections(self, future_data: list, forecast_method: ForecastMethod,
                                    demand_drivers: list, region_name: str) -> pm.Method:
        """
        Process future projections to calculate coefficients using regression.

        :param future_data: List of (year, value) tuples.
        :param forecast_method: Forecasting method to use.
        :param demand_drivers: List of demand drivers.
        :param region_name: Name of the region.
        :return: Coef object with calculated coefficients.
        """
        # Initialize an empty Coef object
        coef = pm.Method()

        # Extract years and values from future_data
        try:
            years, values = zip(*future_data)  # Unpack into separate lists
        except ValueError:
            logging.error("Invalid future_data format. Expected a list of (year, value) tuples.")
            return coef

        years = list(map(int, years))  # Ensure years are integers
        values = list(map(float, values))  # Ensure values are floats

        # Prepare demand driver data for the given years
        demand_driver_data = self._map_filtered_demand_driver_data(
            demand_drivers=demand_drivers, years=years, region_name=region_name, data_origin="user"
        )

        # Convert demand driver data to a 2D array for regression
        demand_driver_array = demand_driver_data.values

        # Perform regression to calculate coefficients
        try:
            coef = self._calculate_coef_for_filtered_data(
                values=values,
                demand_driver_data=demand_driver_array,
                forecast_method=forecast_method
            )
        except Exception as e:
            logging.error(f"Error calculating coefficients for future projections in region '{region_name}': {e}")

        return coef

    def _extract_user_given_coefficients(self, user_data: pd.DataFrame, coef_keys: list) -> list:
        """
        Extract user-provided coefficients from the data and return them as a list.

        :param user_data: DataFrame containing user data.
        :param coef_keys: List of columns representing coefficients (e.g., 'k0', 'k1').
        :return: List of coefficients in the order of coef_keys.
        """
        coefficients = []
        for coef_key in coef_keys:
            try:
                if coef_key in user_data.columns:
                    value = user_data.iloc[0][coef_key]
                    coefficients.append(float(value))
                else:
                    logging.warning(f"Coefficient key {coef_key} not found in user data columns.")
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid or missing value for coefficient {coef_key} in user data: {e}")
                coefficients.append(None)  # Optional: Append None for missing or invalid coefficients
        return coefficients

    def _do_predictions(self, coefficients, forecast_method, demand_drivers, region_name) -> pd.DataFrame:
        """
        Make predictions for a range of years using the specified forecast method.

        :param coefficients: Coef object containing the regression coefficients.
        :param forecast_method: ForecastMethod enum specifying the forecasting method.
        :param demand_drivers: List of demand driver names.
        :param region_name: Name of the region for which prediction is required.
        :return: A DataFrame with predictions.
        """
        start_year = self.general_settings.forecast_year_start
        end_year = self.general_settings.forecast_year_end
        step = self.general_settings.forecast_year_step
        forecast_year_range = range(start_year, end_year + 1, step)

        if hasattr(coefficients, 'forecast_method') and coefficients.forecast_method:
            forecast_method = coefficients.forecast_method

        # Retrieve the prediction function from the forecast map
        method_details = forecast_methods_map[forecast_method]
        predict_function = method_details.get("predict_function")

        if not predict_function:
            raise ValueError(f"Prediction function not defined for forecast method: {forecast_method}")

        # Generate X_values for all years
        predictions = {}
        for year in forecast_year_range:
            try:
                x_values = self.map_x_values(demand_drivers, region_name, year)
                predictions[year] = predict_function(coefficients, x_values)
            except Exception as e:
                logging.error(f"Error predicting for {region_name}, year {year}: {e}")
                predictions[year] = None  # Default to None for errors

        # Structure the DataFrame
        data = {
            "Region": [region_name],
            "Coefficients": [coefficients.coefficients],
            "Equation": [coefficients.equation],
            **predictions
        }
        return pd.DataFrame(data)

    def map_x_values(self, demand_drivers, region_name, year):
        """
        Maps demand drivers to their respective values for a specified region and year.

        :param demand_drivers: List of demand driver names.
        :param region_name: Name of the region.
        :param year: Year for which the values are to be fetched.
        :return: List of values corresponding to the demand drivers for the specified region and year.
        """
        x_values = []
        for driver in demand_drivers:
            if driver == "TIME":
                x_values.append(float(year))
            else:
                data = self.demand_driver_data.get_driver_data(driver)
                if data:
                    region_data = data.get_data_for_region(region_name, "user")
                    if region_data is not None and not region_data.empty:
                        # Extract the value for the specified year
                        region_data.columns = region_data.columns.map(str).str.strip() # Normalize columns
                        if str(year) in region_data.columns:
                            value = region_data[str(year)].iloc[0]
                            x_values.append(float(value) if not pd.isna(value) else np.nan)  # Handle NaN as is
                        else:
                            logging.warning(
                                f"Year {year} not found in data for driver '{driver}' in region '{region_name}'.")
                            x_values.append(np.nan)  # Use NaN for missing years
                    else:
                        logging.warning(f"No data found for region '{region_name}' in driver '{driver}'.")
                        x_values.append(np.nan)  # Use NaN for missing regions
                else:
                    logging.warning(f"No data found for demand driver '{driver}' in region '{region_name}'.")
                    x_values.append(np.nan)  # Use NaN for missing drivers
        return x_values


    def write_sector_predictions_to_excel(self):
        """
        Write predictions to separate Excel files for each sector, with sheets for subsectors.
        """
        output_path = InputManager.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        try:
            for sector_name, sectors_predictions_df_list in self.sectors_predictions.items():
                # Create the full file path for this sector
                sector_file_path = os.path.join(output_path, f"{sector_name}.xlsx")
                with pd.ExcelWriter(sector_file_path, engine="xlsxwriter") as writer:
                    sectors_predictions_df_list.to_excel(writer, sheet_name="Data", index=False)

                print(f"Predictions for sector '{sector_name}' successfully written to {sector_file_path}")
        except Exception as e:
            print(f"Error writing predictions to Excel: {e}")


