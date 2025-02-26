import pandas as pd
import os
import logging
from endemo2.data_structures import prediction_methods as pm
import numpy as np
from endemo2.input_and_settings import input_loader as uty
from endemo2.data_structures.conversions_string import  map_forecast_method_to_string
from endemo2.model_instance.method_map import  forecast_methods_map
from endemo2.data_structures.enumerations import ForecastMethod
from endemo2.input_and_settings.input_sect_sub_var import Sector, Variable, Region,DemandDriver,DemandDriverData

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Forecast:

    def __init__(self, input: list[Sector], demand_driver, input_manager):
        """
        Initialize the Forecast object.
        Args:
            input (list[Sector]): List of Sector objects containing all relevant data.
            demand_driver: demand_drivers instance holding the DDr data for the sector execution
        :return forecast data in the region objects
        """
        self.input = input
        self.demand_drivers = demand_driver
        self.input_manager = input_manager
        self.do_forecast()
        self.write_sector_predictions_to_excel()

    def do_forecast(self):                                                                #TODO to move to Sector class
        """
        Process ECU and DDet variables to prepare coefficients and generate predictions.
        """
        for sector in self.input:
            for subsector in sector.subsectors:
                # Process ECU Variable
                self.forecast_variable(variable = subsector.ecu)
                # Process DDet Variables under Technologies
                for technology in subsector.technologies:
                    for variable in technology.ddets:
                        self.forecast_variable(variable = variable)

    def forecast_variable(self, variable):
        """
        Process a single variable, calculate coefficients, and generate predictions.
        Args:
            variable (Variable): Variable object (ECU or DDet).
        """
        for region_data in variable.region_data:
            if region_data.settings is None or region_data.settings.empty:
                logger.warning(f"No settings for {region_data.name}. Skipping variable: {variable.name}.")
                continue
            # adding forecast of region data
            region_data.forecast = self.forecast_region(region_data)
        variable.consolidate_forecast()


    def forecast_region(self, region_data):
        """
        Generate predictions based on historical or user data.

        Args:
            region_data: object of the Region class storing the data for forecasting and settings
        Returns:
            pd.DataFrame: DataFrame containing the forecast for the region.
        """
        forecast = []
        calc_coef = []  #objects with coef for the region
        interpolations = [] #objects with intp points for the region
        if region_data.historical is not None and not region_data.historical.empty:
            coefficients = self._calc_coeff_hist(region_data) # coefficients: list of coef objetcs
            calc_coef.extend(coefficients)
        if region_data.user is not None and not region_data.user.empty:
            coefficients,interpolation_list = self._process_user(region_data)
            calc_coef.extend(coefficients)
            interpolations.extend(interpolation_list)
        # Generate predictions
        if calc_coef:
            predictions_df = self.do_predictions(calc_coef, region_data)
            forecast.append(predictions_df)
        if interpolations:
            interpolation_df = self.do_interpolation(interpolations, region_data)
            forecast.append(interpolation_df)
        if forecast:
            forecast = pd.concat(forecast, ignore_index=False)
        return forecast

    def _process_user(self, region_data):
        """
        Process user data for a region, identifying coefficients and future projections.
        :return: list of coef objects containing the calculated coefficients or known interpolation points.
        """
        user_data = region_data.user
        region_name = region_data.region_name
        coefficients_list = []  # To store coefficients for all rows
        interpolation_list = []
        # Process each row of user_data
        for index, row in user_data.iterrows():
            row = row.to_frame().T
            row.columns = row.columns.map(str)
            row = uty.clean_dataframe(row)
            # Define a row identifier (e.g., index or other unique attribute)
            if "Type" in row.columns and "Temp_level" in row.columns:
                # Both Type and Temp_level columns exist
                type_identifier = (
                    f"{row['Type'].iloc[0]}_{row['Temp_level'].iloc[0]}"
                    if not pd.isna(row['Type'].iloc[0]) and not pd.isna(row['Temp_level'].iloc[0])
                    else "default"
                )
            elif "Subtech" in row.columns and "Temp_level" not in row.columns:
                # Subtech column exists but Temp_level does not
                type_identifier = (
                    row['Subtech'].iloc[0] if not pd.isna(row['Subtech'].iloc[0]) else "default"
                )
            elif "Type" in row.columns:
                # Only Type column exists
                type_identifier = (
                    row['Type'].iloc[0] if not pd.isna(row['Type'].iloc[0]) else "default"
                )
            else:
                # Neither Type, Temp_level, nor Subtech column exists
                type_identifier = "default"
            forecast_method, demand_drivers, factor, lower_limit = extract_forecast_settings(row)
            # Extract relevant keys for coefficients and years
            coef_keys = [col for col in row.columns if isinstance(col, str) and col.startswith('k')]
            year_keys = [str(col) for col in row.columns if isinstance(col, str) and col.isdigit()]
            # Extract coefficients for the row
            if coef_keys:
                coef = self.extract_user_given_coefficients(row, coef_keys, type_identifier, forecast_method)
                coef.demand_drivers_names = demand_drivers
                coef.factor = factor
                coef.lower_limit = lower_limit
                coefficients_list.append(coef)
            # Extract future projections for the row
            elif year_keys:
                if len(year_keys) > 1:
                    row_future_data = row[year_keys]
                    coef = self.process_future_projections(row_future_data, region_name, forecast_method, demand_drivers, type_identifier)
                    coef.demand_drivers_names = demand_drivers
                    coef.factor = factor
                    coef.lower_limit = lower_limit
                    interpolation_list.append(coef)
                else:
                    coef = pm.Method()
                    coefficient = [row[year_keys].iloc[0].values]
                    coef.name = ForecastMethod.CONST
                    coef.demand_drivers_names = demand_drivers
                    coef.coefficients = coefficient
                    coef.type_identifier = type_identifier
                    coef.factor = factor
                    coef.lower_limit = lower_limit
                    coefficients_list.append(coef)
        return coefficients_list, interpolation_list

    def _calc_coeff_hist(self,region):
        """
        Calculate coefficients from historical data and demand drivers.
        :param region: obj.
        :return: Coefficient object.
        """
        key_columns = ["Type","Temp_level"]
        historical_data = region.historical
        settings = region.settings
        valid_keys = [col for col in key_columns if col in historical_data.columns and col in settings.columns]
        coefficients_list = []  # To store coef objects for all rows that we cna do the predictions
        for index, row in historical_data.iterrows(): #TODO this is looks very wrong
            row = row.to_frame().T
            row.columns = row.columns.map(str)
            settings_row  = settings # in case when we have only 1 settings row
            if valid_keys:
                #extarcting needed settings for processing row:
                merged_df = settings.merge(row, on=valid_keys, how='inner')
                # Identify columns with _x and _y suffix
                merged_df = merged_df.rename(columns=lambda x: x.rstrip('_x'))  # Remove `_x`
                merged_df = merged_df.rename(columns=lambda x: x.rstrip('_y'))  # Remove `_y`
                # Drop duplicate columns (since `_x` and `_y` were merged into a single name)
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                settings_row = uty.clean_dataframe(merged_df) #TODO this contains now both hist nd settings so theoretically we can pass only this
            if "Type" in settings_row.columns and "Temp_level" in settings_row.columns:
                # Both Type and Temp_level columns exist
                type_identifier = (
                    f"{settings_row['Type'].iloc[0]}_{settings_row['Temp_level'].iloc[0]}"
                    if not pd.isna(settings_row['Type'].iloc[0]) and not pd.isna(settings_row['Temp_level'].iloc[0])
                    else "default"
                )
            elif "Subtech" in settings_row.columns and "Temp_level" not in settings_row.columns:
                # Subtech column exists but Temp_level does not
                type_identifier = (
                    settings_row['Subtech'].iloc[0] if not pd.isna(settings_row['Subtech'].iloc[0]) else "default"
                )
            elif "Type" in settings_row.columns:
                # Only Type column exists
                type_identifier = (
                    settings_row['Type'].iloc[0] if not pd.isna(settings_row['Type'].iloc[0]) else "default"
                )
            else:
                # Neither Type, Temp_level, nor Subtech column exists
                type_identifier = "default"
            row = uty.clean_dataframe(row)
            forecast_method, demand_drivers, factor, lower_limit = extract_forecast_settings(settings_row)
            # Separate year columns from non-year columns
            year_columns = [col for col in row.columns if str(col).isdigit()]  # Identify valid year columns
            if not year_columns:
                raise ValueError(f"No valid year columns found in historical data for {region.region_name}.")
            # Ensure year columns are integers
            year_columns_list = list(map(int, year_columns))
            # Set the non-year columns as index temporarily for mapping the DDr data
            row = row[year_columns] # only timeseries data for the row
            # Get filtered demand driver data
            demand_driver_data = self._map_filtered_demand_driver_data(demand_drivers, year_columns_list, region.region_name, #df mapped for historical data coef calculation
                                                                   data_origin="historical")
            # Align historical data with demand driver years
            filtered_years = demand_driver_data.index.tolist()
            common_years = sorted(set(filtered_years).intersection(filtered_years))  # Find common years
            if not common_years:
                raise ValueError(f"No common years between demand drivers and historical data for {region.region_name} with the set {settings_row}")
            common_years = list(map(str, common_years))
            # Filter historical data and demand driver data to only include common years
            historical_values = row[common_years].iloc[0].values.tolist()  # Select region and years
            common_years = list(map(int, common_years))
            demand_driver_array = demand_driver_data.loc[common_years].values  # Align demand driver data numpy arrray
            # Pass the filtered data and aligned demand driver data for coefficient calculation
            coef = self.calculate_coef_for_filtered_data(
                values=historical_values,
                demand_driver_data=demand_driver_array,
                forecast_method=forecast_method,
                type_identifier =type_identifier
             )
            coef.demand_drivers_names = demand_drivers
            coef.factor = factor
            coef.lower_limit = lower_limit
            coefficients_list.append(coef)
        return coefficients_list

    def calculate_coef_for_filtered_data(self, values, demand_driver_data, forecast_method: ForecastMethod, type_identifier) -> pm.Method:
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
            print(f"Warning: Forecast method '{forecast_method}' is not recognized. Using 'CONST_LAST' as fallback.") #TODO new methods are not yet added
            forecast_method = ForecastMethod.CONST_LAST
        coef.name = forecast_method
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
            # Ensure demand_driver_data is a 2D array #case when we will have a single DDr
            if len(demand_driver_data.shape) == 1:
                demand_driver_data = demand_driver_data.reshape(-1, 1)
            # Generate coefficients using the mapped function
            coefficients,equation = generate_coef(X=demand_driver_data, y=values)

            # Save coefficients into the Coef object
            coef.save_coef(coefficients,equation,type_identifier)
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
            driver_data_object = self.demand_drivers.get_driver_data(driver_name)
            if not driver_data_object:
                print(f"Demand driver '{driver_name}' object  not found.")
                aligned_DDr_df[driver_name] = np.nan
                continue
            region_data = driver_data_object.get_data_for_region(region_name, data_origin)
            if region_data is not None and not region_data.empty:
                filtered_values = _filter_values_by_year(region_data, years)
                aligned_DDr_df[driver_name] = filtered_values
            else:
                print(f"No data for demand driver '{driver_name}' in region '{region_name}'.")
                aligned_DDr_df[driver_name] = np.nan
        return aligned_DDr_df

    def process_future_projections(self, row_future_data, region_name, forecast_method, demand_drivers, type_identifier):
        """
        Prepare future data for interpolation
        :param type_identifier: type of processing row
        :param demand_drivers: name of Ddrs
        :param forecast_method: method used for forecast from the settings for the region
        :param region_name: name of processed region
        :param row_future_data: df year,value.
        :return: Coef object with interpolation points.
        """
        # Initialize an empty Coef object
        coef = pm.Method()
        coef.name = forecast_method
        coef.type_identifier = type_identifier
        # Extract years and values from future_data
        years = list(map(int, row_future_data.columns))
        demand_driver_data = self._map_filtered_demand_driver_data(
            demand_drivers=demand_drivers, years=years, region_name=region_name, data_origin="user" #Future DDr projections for intorpalation
        )
        filtered_years = demand_driver_data.index.astype(int)
        common_years = sorted(set(filtered_years).intersection(years))  # Find common years
        if not common_years:
            raise ValueError(
                f"No common years between demand drivers and historical data for {region_name}")
        common_years = list(map(str, common_years))
        # Filter historical data and demand driver data to only include common years
        future_values = row_future_data[common_years].iloc[0].values.tolist()  #  value of the function [f1,f2,...]
        common_years = list(map(int, common_years))
        demand_driver_array = demand_driver_data.loc[common_years].values  # Coordinates of the function [[x1,x2,...],...] 2d array
        known_points = [
            tuple(coord) + (value,)
            for coord, value in zip(demand_driver_array, future_values)
        ]  # list of tuples of known points [(x1,x2,..., f1), ...]
        coef.interp_points = known_points
        return coef

    def extract_user_given_coefficients(self, row: pd.DataFrame, coef_keys: list, type_identifier, forecast_method):
        """
        Extract user-provided coefficients from the data and return them as a list.
        :param forecast_method:
        :param type_identifier:
        :param row: DataFrame containing user data with coef and set.
        :param coef_keys: List of columns representing coefficients (e.g., 'k0', 'k1').
        :return: List of coefficients in the order of coef_keys.
        """
        coef = pm.Method()
        coef.name = forecast_method
        equation = row["Equation"]
        coefficients = []
        for coef_key in coef_keys:
            try:
                if coef_key in row.columns:
                    value = row.iloc[0][coef_key]
                    coefficients.append(float(value))
                else:
                    logging.warning(f"Coefficient key {coef_key} not found in user data columns.")
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid or missing value for coefficient {coef_key} in user data: {e}")

        coef.save_coef(coefficients,equation,type_identifier)
        return coef

    def do_predictions(self, predictions_coef:list, region) -> pd.DataFrame:
        """
        Make predictions for a range of years using the specified forecast method.

        :param predictions_coef: list of Coef object containing the regression coefficients.
        :param region: obj.
        :return: A DataFrame with predictions.
        """
        region_name = region.region_name
        forecast_year_range = self.input_manager.general_settings.get_forecast_year_range()
        region_predictions_df_list = []
        for coef in predictions_coef:
            # Retrieve the prediction function from the forecast map
            method_details = forecast_methods_map[coef.name]
            demand_drivers = coef.demand_drivers_names
            forecast_method = coef.name
            factor = coef.factor
            lower_limit = coef.lower_limit
            predict_function = method_details.get("predict_function")
            if not predict_function:
                raise ValueError(f"Prediction function not defined for forecast method: {forecast_method}")
            # Generate X_values for all years
            predictions = {}
            for year in forecast_year_range:
                try:
                    x_values = self.map_x_values(demand_drivers, region_name, year)
                    predicted_value = predict_function(coef, x_values)
                    if predicted_value < lower_limit:
                        predictions[year] = lower_limit
                    else:
                        predictions[year] = predicted_value
                except Exception as e:
                    logging.error(f"Error predicting for {region_name}, year {year}: {e}")
                    predictions[year] = pd.NaT  # Default to NaN for errors
            # Structure the DataFrame
            scaled_predictions = {year: value * factor for year, value in predictions.items()} #scale by factor to get standard values
            data = {
                "Region": [region_name],
                "Coefficients/intp_points": [coef.coefficients],
                "Equation": [coef.equation],
                "Type": [coef.type_identifier],
                "Factor": [factor],
                **scaled_predictions
            }
            region_predictions_df_list.append(pd.DataFrame(data))
        return pd.concat(region_predictions_df_list, ignore_index=False)

    def do_interpolation(self, interpolations, region) -> pd.DataFrame:
        """
        do interpolations for a range of years using the specified forecast method.
        :param interpolations: list of Coef object containing the knowk interpolation obj.
        :param region: obj.
        :return: A DataFrame with predictions.
        """
        region_name = region.region_name
        forecast_year_range = self.input_manager.general_settings.get_forecast_year_range()
        region_interpolations_df_list = []
        for coef in interpolations:
            # Retrieve the prediction function from the forecast map
            method_details = forecast_methods_map[coef.name]
            demand_drivers = coef.demand_drivers_names
            factor = coef.factor
            forecast_method = coef.name
            lower_limit = coef.lower_limit
            predict_function = method_details.get("predict_function")
            if not predict_function:
                raise ValueError(f"Prediction function not defined for forecast method: {forecast_method}")
            # Generate X_values for all years
            predictions = {}
            interp_points = coef.interp_points
            for year in forecast_year_range:
                try:
                    x_values = self.map_x_values(demand_drivers, region_name, year) #points where the interp are needed
                    predicted_value = predict_function(interp_points, x_values)
                    if predicted_value <= lower_limit:
                        predictions[year] = lower_limit
                    else:
                        predictions[year] = predicted_value
                except Exception as e:
                    logging.error(f"Error interpolating for {region_name}, {region.settings} , year {year}: {e}")
                    predictions[year] = pd.NaT # Default to NaN for errors suche when the dot is out of interp boundarries
            # Structure the DataFrame
            scaled_predictions = {year: value * factor for year, value in predictions.items()}
            data = {
                "Region": [region_name],
                "Coefficients/intp_points": [interp_points],
                "Equation": [coef.equation],
                "Type": [coef.type_identifier],
                **scaled_predictions
            }
            region_interpolations_df_list.append(pd.DataFrame(data))
        return pd.concat(region_interpolations_df_list, ignore_index=False)

    def map_x_values(self, demand_drivers, region_name, year):
        """
        Maps demand drivers to their respective values for a specified region and year in future.

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
                data = self.demand_drivers.get_driver_data(driver)
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
        Write predictions to separate Excel files for each sector,
        concatenating all forecast data into a single table per sector.
        """
        output_path = self.input_manager.output_path
        os.makedirs(output_path, exist_ok=True)

        sector_predictions = {}  # Dictionary to store predictions for each sector

        for sector in self.input:
            sector_file_path = os.path.join(output_path, f"predictions_{sector.name}.xlsx")

            # Create a master DataFrame for the sector
            sector_data = []

            for subsector in sector.subsectors:
                for variable, technology_name in [(subsector.ecu, None)] + [
                    (var, tech.name) for tech in subsector.technologies for var in tech.ddets
                ]:
                    for region in variable.region_data:
                        df = region.forecast
                        if df is not None or not df:
                            # Add identifying columns for clarity
                            df = df.copy()  # Avoid modifying the original data
                            df['Subsector'] = subsector.name
                            df['Variable'] = variable.name
                            df['Technology'] = technology_name
                            columns_order = ["Region",'Subsector', 'Variable',"Technology","Type"] + [col for col in df.columns if
                                                                         col not in ["Region",'Subsector', 'Variable', "Technology","Type"]]
                            df = df[columns_order]
                            # Append the DataFrame to the sector data list
                            sector_data.append(df)
            # Concatenate all DataFrames into one
            if sector_data:
                combined_df = pd.concat(sector_data, ignore_index=True)
                sector_predictions[sector.name] = combined_df
                # Write to Excel
                with pd.ExcelWriter(sector_file_path, engine="xlsxwriter") as writer:
                    combined_df.to_excel(writer, sheet_name="Sector Forecast", index=False)
                print(f"Prediction for sector '{sector.name}' was written to: {sector_file_path}")

        return

def extract_forecast_settings(settings_data):
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
    factor = (
        settings_data.iloc[0]['Factor']
        if 'Factor' in settings_data.columns else None
    )
    lower_limit = float((
        settings_data.iloc[0]['Lower limit']
        if 'Lower limit' in settings_data.columns else None
    ))
    return forecast_method, demand_drivers, factor, lower_limit

def _filter_values_by_year(region_data: pd.DataFrame, years: list) -> list:
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

def _extract_future_projections(row: pd.DataFrame, year_keys: list):
    """
    Extract future projections from the user data.

    :param row: DataFrame containing user data.
    :param year_keys: List of columns representing future years.
    :return: List of tuples (year, value) for future projections.
    """
    projections = []
    for year in year_keys:
        try:
            value = float(row.iloc[0][year])
            projections.append((int(year), value))
        except ValueError:
            logging.warning(f"Invalid value for year {year} in user data.")
    return projections
