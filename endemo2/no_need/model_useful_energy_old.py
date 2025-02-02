import pandas as pd
from endemo2.input_and_settings.input_sect_sub_var import  UsefulenergyRegion
import os

class UsefulEnergyCalculator:
    def __init__(self, input_manager, input):
        """
        Initialize the UsefulEnergyCalculator.
        :param input_manager: InputManager instance providing access to settings and subsector configurations.
        :param input: list of sector objects that were processed in the forecast model.
        """
        self.input_manager = input_manager
        self.predictions = input
        self.all_sectors_UE = {}  # To store results  by sector
        self.calculate_useful_energy()
        self.write_ue_predictions_to_excel()

    def calculate_useful_energy(self):
        """
        Calculate useful energy as ECU * sum(DDet[Technology][Type]) for each subsector,
        for each combination of DDet, Technology, and Type.
        """
        # Initialize the dictionary to store UsefulenergyRegion objects by region
        region_ue_dict = {}
        general_settings = self.input_manager.general_settings
        active_regions = self.input_manager.ctrl.GEN_settings.active_regions
        forecast_year_range = general_settings.get_forecast_year_range()
        active_regions = [region for region in active_regions if region != "default"]
        for sector in self.predictions:
            sector_name = sector.name
            ue_per_sector = []
            for subsector in sector.subsectors:
                # Gather region-level results for this subsector
                ecu = subsector.ecu
                technologies = subsector.technologies
                subsector_name = subsector.name
                ue_per_subsectors = []
                for technology in technologies:
                    technology_name = technology.name
                    ddets = technology.ddets # list of dddet variables per technology
                    ue_per_technology_for_all_regions = self.calculate_ue_regios(ecu, ddets, active_regions,
                                                                                 forecast_year_range, region_ue_dict, subsector_name,
                                                                                 technology_name, sector_name)
                    technology.energy_UE = ue_per_technology_for_all_regions
                    ue_per_subsectors.append(ue_per_technology_for_all_regions) #adding  the technology to the subsector that it belongs
                ue_per_subsector = pd.concat(ue_per_subsectors, ignore_index=True)
                subsector.energy_UE = ue_per_subsector #TODO i dont know why i am saving each co
                ue_per_sector.append(ue_per_subsector)
            ue_sector = pd.concat(ue_per_sector, ignore_index=True)
            sector.energy_UE = ue_sector # now sector holds all regions which is not completly right
        self.export_ue_per_region_data(region_ue_dict)

    def calculate_ue_regios(self, ecu, ddets, active_regions, forecast_year_range, region_ue_dict,
                            subsector_name, technology_name, sector_name):
        """
        Calculate useful energy for each Technology dynamically.
        :param ecu: The ecu of the region
        :param ddets: list of subsector's technology's ddets
        :param active_regions: list of regions for calculations.
        :param forecast_year_range: List of forecast years.
        :return: DataFrame containing useful energy calculations for the sebusector's technology.
        """
        # Ensure forecast_year_range is in string format
        ddets_dfs = []
        variable_names = []
        variable_names.append(ecu.name)
        for ddet in ddets:
            df = ddet.forecast
            ddet_name = ddet.name
            ddets_dfs.append(df)
            variable_names.append(ddet_name)
        forecast_year_range = [str(year) for year in forecast_year_range]
        ue_per_technology_for_all_regions = [] #this is beacuse of the structure
        for region in active_regions:
            # Check if the region is already initialized in region_ue_list
            if region not in region_ue_dict: #holds the objects that collect the ue per region for all sectors
                region_ue_dict[region] = UsefulenergyRegion(name=region)
            region_ue = region_ue_dict[region]
            # Extract ECU row
            ecu_row = ecu.forecast[ecu.forecast["Region"] == region]
            ecu_row.columns = ecu_row.columns.map(str)
            ecu_values = ecu_row[forecast_year_range]
            list_ddets_for_region = [df[df["Region"]  == region ] for df in ddets_dfs]
            #separate with type and without
            dfs_with_type = [df[df["Type"] != "default"] for df in list_ddets_for_region]
            dfs_without_type = [df[df["Type"] == "default"] for df in list_ddets_for_region]
            dfs_without_type= [df for df in dfs_without_type if not df.empty]
            dfs_with_type = [df for df in dfs_with_type if not df.empty]

            dfs_with_shared_type = []  # List to store extracted HEAT_Q DataFrames
            dfs_with_total_type = []  # List to store DataFrames without HEAT_Q

            for df in dfs_with_type:
                heat_q_df = df[df["Type"].str.startswith("HEAT_Q", na=False)]  # Extract HEAT_Q rows
                remaining_df = df[~df["Type"].str.startswith("HEAT_Q", na=False)]  # Keep the rest
                if not heat_q_df.empty:
                    dfs_with_shared_type.append(heat_q_df)  # Add the HEAT_Q DataFrame to the list
                if not remaining_df.empty:
                    dfs_with_total_type.append(remaining_df)

            # Initialize the common multiplier
            common_multiplier = pd.Series(1, index=forecast_year_range)
            # Compute the common multiplier from default rows and ECU values
            for variable in dfs_without_type + [ecu_values]:
                if variable is None or variable.empty:
                    continue
                variable.columns = variable.columns.map(str)
                common_multiplier *= variable[forecast_year_range].iloc[0]
            # Process rows with specific Types
            combined_with_type = pd.concat(dfs_with_total_type, ignore_index=True)
            grouped_by_type = combined_with_type.groupby("Type")  # Group by 'Type'
            aggregated_results = pd.Series(0, index=forecast_year_range) # (aggregated by types)
            ue_types = [] # this is ue for the region per types for 1 technology for 1 subsector for 1 sector
            type_names = []
            for type_name, group in grouped_by_type:
                type_names.append(type_name)
                group.columns = group.columns.map(str)
                # Compute the product of year columns within the group
                group_product = group[forecast_year_range].prod(axis=0)
                # Multiply with the common multiplier
                results_ue_type = group_product * common_multiplier #pd.series type
                # Create a result row
                result_row = {"Region": region,
                              "Sector": sector_name,
                              "Subsector": subsector_name,
                              "Technology": technology_name,
                              "Type": type_name,
                              "Variables": variable_names}
                result_row.update(results_ue_type.to_dict())
                ue_types.append(result_row)
                if type_name == "HEAT" and not all(df.empty for df in dfs_with_shared_type):
                    combined_with_temp_level = pd.concat(dfs_with_shared_type, ignore_index=True)
                    grouped_by_level = combined_with_temp_level.groupby("Type")
                    for temp_level, group_per_level in grouped_by_level:
                        group_per_level.columns = group_per_level.columns.map(str)
                        prod_per_level = group_per_level[forecast_year_range].prod(axis=0)
                        share_per_level = prod_per_level * results_ue_type
                        temp_level_row = {"Region": region,
                                      "Sector": sector_name,
                                      "Subsector": subsector_name,
                                      "Technology": technology_name,
                                      "Type": temp_level,
                                      "Variables": variable_names}
                        temp_level_row.update(share_per_level.to_dict())
                        ue_types.append(temp_level_row)
                # Add the results_ue_type to the aggregated total per types thus we get per technology
                aggregated_results += results_ue_type
            # Combine results into a DataFrame
            ue_results_per_types = pd.DataFrame(ue_types)
            # here we collect the ue for the region for all sectors
            region_ue.energy_ue.append(ue_results_per_types)
            ue_per_technology_for_all_regions.append(ue_results_per_types)


            # Add the results_ue_type to the aggregated total per types thus we get per technology #TODO in the new structure when the region is on the top level
            # aggregated_row = {
            #     "Region": region,
            #     "Sector": sector_name,
            #     "Subsector": subsector_name,
            #     "Technology": technology_name,
            #     "Type": type_names,
            #     "Variables": f"Aggregated per type {variable_names}"
            # }
            # aggregated_row.update(aggregated_results.to_dict())


        return  pd.concat(ue_per_technology_for_all_regions, ignore_index=True)

    def write_ue_predictions_to_excel(self):
        """
        Write predictions to separate Excel files for each sector,
        concatenating all forecast data into a single table per sector.
        """
        output_path = self.input_manager.output_path
        os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists

        for sector in self.predictions:
            # Construct the file path for the sector
            sector_file_path = os.path.join(output_path, f"ue_{sector.name}.xlsx")
            # Access the sector.energy_UE DataFrame
            if hasattr(sector, "energy_UE") and isinstance(sector.energy_UE, pd.DataFrame):
                with pd.ExcelWriter(sector_file_path, engine="xlsxwriter") as writer:
                    sector.energy_UE.to_excel(writer, sheet_name="Energy_UE", index=False)
                print(f"Predictions of useful energy for sector '{sector.name}' written to {sector_file_path}")
            else:
                print(f"Skipping sector '{sector.name}' - 'energy_UE' is not a valid DataFrame or missing.")

    def export_ue_per_region_data(self, region_ue_dict, output_folder="region_ue"):
        """
        Export the useful energy data for all regions to Excel after all calculations are complete.

        :param region_ue_dict: Dictionary containing UsefulenergyRegion objects keyed by region name.
        :param output_folder: Folder where the Excel files should be saved (relative to input_manager.output_path).
        """
        # Use the output path from InputManager and append the output_folder
        full_output_path = self.input_manager.output_path / output_folder
        # Ensure the full output folder exists
        os.makedirs(full_output_path, exist_ok=True)
        # Export each region's data
        for region_name, region_ue in region_ue_dict.items():
            region_ue.export_to_excel(full_output_path)



