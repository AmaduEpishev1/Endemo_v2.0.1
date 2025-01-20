import pandas as pd
import os

class UsefulEnergyCalculator:
    def __init__(self, input_manager, predictions):
        """
        Initialize the UsefulEnergyCalculator.

        :param input_manager: InputManager instance providing access to settings and subsector configurations.
        :param predictions: Nested dictionary containing prediction results for ECU and DDet variables.
        """
        self.input_manager = input_manager
        self.predictions = predictions
        self.all_sectors_UE = {}  # To store results  by sector
        self.calculate_useful_energy()
        self.save_results_to_excel()

    def calculate_useful_energy(self):
        """
        Calculate useful energy as ECU * product(DDet[Technology][Type]) for each subsector,
        for each combination of DDet, Technology, and Type.
        """
        subsector_settings = self.input_manager.subsector_settings.subsector_settings
        general_settings = self.input_manager.general_settings
        active_regions = self.input_manager.ctrl.GEN_settings.active_regions

        forecast_year_range = list(range(
            general_settings.forecast_year_start,
            general_settings.forecast_year_end + 1,
            general_settings.forecast_year_step,
        ))

        for sector_name, sector_df in self.predictions.items():
            # List to hold all subsector DataFrames for the current sector
            sector_results = []

            # Identify unique subsectors within the sector DataFrame
            unique_subsectors = sector_df['Subsector'].unique()

            for subsector_name in unique_subsectors:
                # Extract DataFrame for the current subsector
                subsector_predictions = sector_df[sector_df['Subsector'] == subsector_name].copy()
                # Retrieve settings for the current subsector
                settings = subsector_settings.get(f"{sector_name}_subsectors")
                ecu_name, ddet_names, _ = self._extract_ecu_ddet_and_tech(settings, subsector_name)
                # Normalize the DataFrame
                normalized_subsector_df = self.normalize_df(subsector_predictions, ddet_names)

                # Gather region-level results for this subsector
                subsector_results = []
                for region in active_regions:
                    region_group_df = normalized_subsector_df[normalized_subsector_df["Region"] == region]
                    results_UE_region = self._process_region_data(
                        region, region_group_df, ecu_name, ddet_names, forecast_year_range,
                        sector_name, subsector_name
                    )
                    if not results_UE_region.empty:
                        results_UE_region = results_UE_region.drop(columns=['Variable',"Coefficients","Equation"])
                        subsector_results.append(results_UE_region)

                # Concatenate all region-level results for the subsector
                if subsector_results:
                    subsector_UE_df = pd.concat(subsector_results, ignore_index=True)
                    sector_results.append(subsector_UE_df)

            # Concatenate all subsector DataFrames for the current sector
            if sector_results:
                sector_UE_df = pd.concat(sector_results, ignore_index=True)
                self.all_sectors_UE[sector_name] = sector_UE_df

    def save_results_to_excel(self):
        """
        Save results to Excel files, one per sector, with a single sheet for each sector.

        """
        output_path = self.input_manager.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for sector_name, sector_df in self.all_sectors_UE.items():
            column_order = ["Region", "Sector", "Subsector", "Technology","Type", "Variables"] + [
                col for col in sector_df.columns if
                col not in ["Region", "Sector", "Subsector", "Technology","Type", "Variables"]   ]
            sector_df = sector_df[column_order]
            # Construct the file name based on the sector name
            file_name = os.path.join(output_path, f"UE_{sector_name}.xlsx")
            # Save the sector DataFrame to an Excel file
            sector_df.to_excel(file_name, index=False, sheet_name="Data")
            print(f"Results saved for sector: {sector_name} in file: {file_name}")

    def _process_region_data(self, region, region_group_df, ecu_name, ddet_names, forecast_year_range, sector_name,
                             subsector_name):
        """
        Calculate useful energy for each Technology dynamically.

        :param region: The region name for which to calculate useful energy.
        :param region_group_df: DataFrame containing data for the specific region and subsector.
        :param ecu_name: Name of the ECU variable.
        :param ddet_names: List of DDet variable names.
        :param forecast_year_range: List of forecast years.
        :param sector_name: Name of the sector.
        :param subsector_name: Name of the subsector.
        :return: DataFrame containing useful energy calculations for the region.
        """
        # Ensure forecast_year_range is in string format
        forecast_year_range = [str(year) for year in forecast_year_range]
        region_group_df.columns = region_group_df.columns.map(str)

        # Extract ECU row
        ecu_row = region_group_df[region_group_df["Variable"] == ecu_name]
        if ecu_row.empty:
            print(f"No ECU row found for region: {region}")
            return pd.DataFrame()

        # Extract ECU values
        try:
            ecu_values = ecu_row.iloc[0][forecast_year_range].values
        except KeyError as e:
            print(f"Error accessing forecast year columns: {e}")
            return pd.DataFrame()
        # Remove ECU row from the main DataFrame to avoid inclusion in Technology processing
        region_group_df = region_group_df[region_group_df["Variable"] != ecu_name]
        # Filter by Technology
        results = []
        for technology in region_group_df["Technology"].dropna().unique():
            filtered_df_ddet = region_group_df[region_group_df["Technology"] == technology]

            # Compute DDet product for the filtered DataFrame
            ddet_product_df = self.compute_Ddet_product(filtered_df_ddet, ddet_names,forecast_year_range, ecu_name)

            # Multiply ECU values with the DDet product for each year
            for i, year in enumerate(forecast_year_range):
                ddet_product_df[year] *= ecu_values[i]

            # Append to results
            results.append(ddet_product_df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def compute_Ddet_product(self, filtered_df_Ddet, ddet_names, forecast_year_range,ecu_name):
        """
        Compute the product of DDet values for the given forecast years, considering rows with and without Type.

        Parameters:
            filtered_df_Ddet (DataFrame): Filtered DataFrame for a specific Technology.
            ddet_names (list): List of DDet variable names to consider for multiplication.
            forecast_year_range (list): List of forecast years.

        Returns:
            DataFrame: DataFrame containing updated forecast values for each combination, grouped by Type.
        """
        ddet_product = []
        # Filter rows corresponding to DDet variables
        ddet_rows = filtered_df_Ddet[filtered_df_Ddet["Variable"].isin(ddet_names)]
        if ddet_rows.empty:
            print("No DDet data found.")
            return pd.DataFrame(columns=filtered_df_Ddet.columns)  # Return empty DataFrame if no valid rows

        # Separate rows with and without Type
        ddet_with_type = ddet_rows[ddet_rows["Type"].notna()]
        ddet_without_type = ddet_rows[ddet_rows["Type"].isna() | (ddet_rows["Type"] == "")]  # Include empty strings


        # Start with rows without a Type (common multiplier for all Types)
        common_multiplier = pd.Series(1.0, index=forecast_year_range)
        for _, row in ddet_without_type.iterrows():
            common_multiplier *= row[forecast_year_range]

        # Iterate over each unique Type
        unique_types = ddet_with_type["Type"].unique()
        for type_value in unique_types:
            # Filter rows for the current Type
            type_filtered_rows = ddet_with_type[ddet_with_type["Type"] == type_value]

            # Start with the product of rows for the current Type
            base_product = pd.Series(1.0, index=forecast_year_range)

            for _, row in type_filtered_rows.iterrows():
                base_product *= row[forecast_year_range]

            # Multiply the common multiplier into the base product
            base_product *= common_multiplier

            # Prepare the row for the product of rows with this Type
            result_row = type_filtered_rows.iloc[0].copy()  # Use metadata from the first row of the current Type
            result_row[forecast_year_range] = base_product
            result_row["Variables"] = f"Product({ddet_names + [ecu_name]})"  # name of the Variables which are multipled
            ddet_product.append(result_row)

        # Convert the result into a DataFrame
        if ddet_product:
            return pd.DataFrame(ddet_product)
        else:
            return pd.DataFrame(columns=filtered_df_Ddet.columns)

    def _extract_ecu_ddet_and_tech(self, settings, subsector_name):
        """
        Extract ECU, DDet variable names, and Technology for the specified subsector from the settings table.

        :param settings: Settings DataFrame.
        :param subsector_name: Name of the subsector to filter settings.
        :return: Tuple of ECU name, list of DDet names, and list of technologies (if applicable).
        """
        # Filter the settings table for the specified subsector
        filtered_row = settings[settings["Subsector"] == subsector_name]
        if filtered_row.empty:
            print(f"No settings found for subsector: {subsector_name}")
            return None, [], []

        # Extract ECU
        ecu_name = filtered_row["ECU"].iloc[0] if "ECU" in filtered_row.columns else None

        # Extract DDet variables
        ddet_columns = [col for col in filtered_row.columns if col.startswith("DDet")]
        ddet_names = [filtered_row[col].iloc[0] for col in ddet_columns if not pd.isna(filtered_row[col].iloc[0])]

        # Extract Technology
        technologies = (
            filtered_row["Technology"].iloc[0].split(",") if "Technology" in filtered_row.columns and not pd.isna(
                filtered_row["Technology"].iloc[0]) else []
        )

        return ecu_name, ddet_names, technologies

    def normalize_df(self, subsector_df, ddet_names):
        """
        Splits the Variable column in the DataFrame into two separate columns: Variable and Type.
        :param subsector_df: Input DataFrame containing the Variable column.
        :param ddet_names: List of base variable names to match against the Variable column.
        :return: Transformed DataFrame with Variable and Type columns.
        """

        # Function to extract base variable and type
        def extract_variable_and_type(var_name):
            """
            Extracts the base variable and the remaining part (type).
            If the remaining part (type) is empty, it will be replaced with None.
            :param var_name: The variable name string to process.
            :return: Tuple of (base_key, type_part) or (var_name, None) if no match.
            """
            # Match the base variable name from the defined keys
            for base_key in ddet_names:
                if base_key in var_name:
                    # Extract the base variable and the remaining part as the type
                    type_part = var_name.replace(base_key, "").strip()
                    # Replace empty type_part with None
                    type_part = type_part if type_part else None
                    return base_key, type_part
            # If no match, return the full name as Variable and Type as None
            return var_name, None

        # Apply the extraction function to the Variable column
        subsector_df[["Variable", "Type"]] = subsector_df["Variable"].apply(
            lambda x: pd.Series(extract_variable_and_type(x))
        )
        return subsector_df


# class UsefulEnergyAggregator:
#     def _init_(self,UsefulEnergyCalculator):
#          sectors_ue = UsefulEnergyCalculator.all_sectors_UE
#
#     def aggregatre_by_region_for_technology(self):
#         pass