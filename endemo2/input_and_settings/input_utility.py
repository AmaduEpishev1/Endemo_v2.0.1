import warnings
from pathlib import Path

import pandas as pd
from endemo2.input_and_settings.input_sect_sub_var import Sector,Subsector,Variable,Region,Technology



def initialize_hierarchy_input_from_settings_data(active_sectors, subsector_settings, input_manager):
    sectors = []  # Initialize a list to hold sectors

    for sector_name in active_sectors:
        sector = Sector(name=sector_name)
        key = f"{sector_name}_subsectors"
        if key not in subsector_settings:
            print(f"No data found for sector: {sector_name}")
            continue
        subsectors_df = subsector_settings[key]
        for _, row in subsectors_df.iterrows():
            if not row["Active"]:
                continue
            subsector_name = row["Subsector"]
            # Get "Technology" column if it exists
            if "Technology" in row and pd.notna(row["Technology"]):
                technology_list = [tech.strip() for tech in row["Technology"].split(",")]
            else:
                technology_list = ["default"]

            # Initialize Subsector
            subsector = Subsector(name=subsector_name)

            # Process ECU Variables
            ecu = row.get("ECU")
            if ecu:
                regions = load_region_data_for_ECU_variable(input_manager, sector_name, row["Subsector"], ecu)
                variable = Variable(name=ecu)
                for region in regions:
                    variable.region_data.append(region)
                subsector.ecu = variable # Add to Subsector

            # Process DDet Variables
            ddet_columns = {col: row[col] for col in row.index if col.startswith("DDet") and pd.notna(row[col])}
            for ddet_name in ddet_columns.values():
                technology_objects = load_region_data_for_DDet_variable(
                    input_manager, sector_name, subsector_name, ddet_name, technology_list
                )
                for tech in technology_objects:
                    subsector.technologies.append(tech)

            # Add Subsector to Sector
            sector.subsectors.append(subsector)

        # Add Sector to the list of sectors
        sectors.append(sector)

    return sectors  # Return the list of sectors

def clean_dataframe(df):
    if df is None:
        return df
    # Drop rows/columns with all NaN values
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    return df

def load_region_data_for_DDet_variable(input_manager, sector_name, subsector_name, ddet_name, technology_list):
    """
    Load region data for a DDet variable dynamically, even if Type or Temp_level columns are missing.
    """
    technology_objects = []  # List of Technology objects
    active_regions = input_manager.ctrl.GEN_settings.active_regions

    # Paths to settings and data files
    settings_path = input_manager.settings_paths[sector_name] / f"{ddet_name}_set.xlsx"
    historical_path = input_manager.historical_data_paths[sector_name] / f"{ddet_name}_hist.xlsx"
    user_path = input_manager.user_forecast_data_paths[sector_name] / f"{ddet_name}_user.xlsx"

    for technology_name in technology_list:
        sheet_name = f"{subsector_name}_{technology_name}" if technology_name != "default" else subsector_name
        technology = Technology(name=technology_name)  # Create Technology object

        try:
            # Load settings data
            settings_df = pd.read_excel(settings_path, sheet_name=sheet_name)
            settings_df = clean_dataframe(settings_df)

            # Check if Type or Temp_level columns exist
            has_type = "Type" in settings_df.columns
            has_temp_level = "Temp_level" in settings_df.columns

            # Group by available columns
            if has_type and has_temp_level:
                grouped_settings = settings_df.groupby(["Type", "Temp_level"], dropna=False)
            elif has_type:
                grouped_settings = settings_df.groupby("Type", dropna=False)
            else:
                grouped_settings = [("default", settings_df)]

            for group_key, settings_group in grouped_settings:
                # Handle dynamic keys for Type and Temp_level
                if has_type and has_temp_level:
                    type_val, temp_level = group_key # This works when group_key is a tuple of 2
                elif has_type:
                    type_val = group_key # group_key is now a single value, no unpacking needed
                    temp_level = None
                else:
                    type_val, temp_level = "default", None

                for region_name in active_regions:
                    region_settings = settings_group[settings_group["Region"] == region_name]
                    if region_settings.empty:
                        continue

                    # Determine Forecast data source (User or Historical)
                    forecast_data = region_settings.iloc[0].get("Forecast data", None)
                    data_df = None
                    source_name = None

                    if forecast_data == "User":
                        if user_path.exists():
                            data_df = pd.read_excel(user_path, sheet_name=sheet_name)
                            data_df = clean_dataframe(data_df)
                            source_name = "user"
                    elif forecast_data == "Historical":
                        if historical_path.exists():
                            data_df = pd.read_excel(historical_path, sheet_name=sheet_name)
                            data_df = clean_dataframe(data_df)
                            source_name = "historical"

                    # Filter data for the current Type, Temp_level, and Region
                    if has_type and has_temp_level:
                        filtered_data = data_df[
                            (data_df["Region"] == region_name) &
                            (data_df["Type"] == type_val) &
                            ((data_df["Temp_level"] == temp_level) | pd.isna(temp_level))
                        ]
                    elif has_type:
                        filtered_data = data_df[
                            (data_df["Region"] == region_name) &
                            (data_df["Type"] == type_val)
                        ]
                    else:
                        filtered_data = data_df[data_df["Region"] == region_name]
                        filtered_data=clean_dataframe(filtered_data)

                    if filtered_data.empty:
                        continue

                    # Create RegionData and add type data
                    region = Region(region_name, settings=region_settings)
                    temp_level_key = "" if pd.isna(temp_level) else temp_level
                    # Add type data with or without temp_level
                    if temp_level_key:  # If temp_level exists
                        region.add_type_data(f"{type_val}_{temp_level_key}", filtered_data, source_name)
                    else:  # If temp_level does not exist
                        region.add_type_data(f"{type_val}", filtered_data, source_name)

                    # Extract DDr values
                    region.extract_ddr_values()
                    input_manager.global_ddr_list.update(region.ddr_values)

                    # Add the region data to a variable object
                    variable_name = f"{ddet_name}_{type_val}_{temp_level_key}" if has_type else f"{ddet_name}"
                    variable = Variable(name=variable_name)
                    variable.region_data.append(region)
                    technology.ddets.append(variable)

        except Exception as e:
            print(f"Error processing DDet variable {ddet_name} for technology {technology_name}: {e}")

        technology_objects.append(technology)

    return technology_objects


def load_region_data_for_ECU_variable(input_manager, sector_name, subsector_name, variable_value):
    """
    Dynamically load region data for a given ECU variable in a subsector.

    :param input_manager: InputManager instance to access file paths and settings.
    :param sector_name: The name of the sector.
    :param subsector_name: The name of the subsector.
    :param variable_value: The ECU variable.
    :return: List of RegionData objects for the variable.
    """
    regions = []
    active_regions = input_manager.ctrl.GEN_settings.active_regions
    settings_path = input_manager.settings_paths[sector_name] / f"{variable_value}_set.xlsx"

    try:
        # Load and clean settings DataFrame
        settings_df = pd.read_excel(settings_path, sheet_name=subsector_name)
        settings_df = clean_dataframe(settings_df)

        for region_name in active_regions:
            # Filter settings for the region
            region_settings = settings_df[settings_df["Region"] == region_name]
            region_settings = clean_dataframe(region_settings)
            if region_settings.empty:
                print(f"No settings for region: {region_name}. Skipping...")
                continue

            # Determine forecast data type
            forecast_data = region_settings.iloc[0].get("Forecast data", "Historical")
            data_path = None
            if forecast_data == "Historical":
                data_path = input_manager.historical_data_paths[sector_name] / f"{variable_value}_hist.xlsx"
            elif forecast_data == "User":
                data_path = input_manager.user_forecast_data_paths[sector_name] / f"{variable_value}_user.xlsx"

            # Initialize data variables
            historical_data, user_data = None, None

            # Load and clean data files
            if data_path and data_path.exists():
                data_df = pd.read_excel(data_path, sheet_name=subsector_name)
                data_df = clean_dataframe(data_df)

                # Filter data for the current region
                region_data_df = data_df[data_df["Region"] == region_name]
                region_data_df  = clean_dataframe(region_data_df)
                if not region_data_df.empty:
                    if forecast_data == "Historical":
                        historical_data = region_data_df
                    elif forecast_data == "User":
                        user_data = region_data_df
                else:
                    print(f"No data found for region: {region_name} in {forecast_data} file.")

            else:
                print(f"Data file not found: {data_path}. Skipping region: {region_name}.")

            # Create and add RegionData object
            region = Region(
                region_name=region_name,
                settings=region_settings
            )

            # Use add_type_data to align with DDet structure
            if historical_data is not None and not historical_data.empty:
                region.add_type_data("default", historical_data, "historical")

            if user_data is not None and not user_data.empty:
                region.add_type_data("default", user_data, "user")

            # Extract DDr values and update the global list
            region.extract_ddr_values()
            input_manager.global_ddr_list.update(region.ddr_values)

            # Append the region to the list
            regions.append(region)

    except Exception as e:
        print(f"Error loading settings data for {variable_value} in {subsector_name}: {e}")

    return regions

def skip_years_in_df(df: pd.DataFrame, skip_years: [int]):
    for skip_year in skip_years:
        if skip_year in df.columns:
            df.drop(skip_year, axis=1, inplace=True)



class FileReadingHelper:
    """
    A helper class to read products historical data. It provides some fixed transformation operations.

    :ivar str file_name: The files name, relative to the path variable.
    :ivar str sheet_name: The name of the sheet that is to be read from the file.
    :ivar [int] skip_rows: These rows(!) will be skipped when reading the dataframe. Done by numerical index.
    :ivar lambda[pd.Dataframe -> pd.Dataframe] sheet_transform: A transformation operation on the dataframe
    :ivar pd.Dataframe df: The current dataframe.
    :ivar Path path: The path to the folder, where the file is.
        It can to be set after constructing the FileReadingHelper Object.
    """

    def __init__(self, file_name: str, sheet_name: str, skip_rows: [int], sheet_transform):
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.skip_rows = skip_rows
        self.sheet_transform = sheet_transform
        self.df = None
        self.path = None

    def set_path_and_read(self, path: Path) -> None:
        """
        Sets the path variable and reads the file with name self.file_name in the path folder.

        :param path: The path, where the file lies.
        """
        self.path = path
        self.df = self.sheet_transform(pd.read_excel(self.path / self.file_name, self.sheet_name,
                                                     skiprows=self.skip_rows))

    def skip_years(self, skip_years: [int]) -> None:
        """
        Filters the skip years from the current dataframe.

        :param skip_years: The list of years to skip.
        """
        if self.df is None:
            warnings.warn("Trying to skip years in products historical data without having called set_path_and_read"
                          " on the Retrieve object.")
        skip_years_in_df(self.df, skip_years)

    def get(self) -> pd.DataFrame:
        """
        Getter for the dataframe.

        :return: The current dataframe, which is filtered depending on previous function calls on this class.
        """
        if self.df is None:
            warnings.warn("Trying to retrieve products historical data without having called set_path_and_read on "
                          "the Retrieve object.")
        return self.df





