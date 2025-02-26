import pandas as pd
from endemo2.input_and_settings.input_sect_sub_var import Sector,Subsector,Variable,Region,Technology,DemandDriver

def initialize_hierarchy_input_from_settings_data(input_manager):
    """
       Initialize hierarchy input using settings data for active sectors.

       Parameters:
           input_manager : obj that contain all setting parameters for model execution.
       Returns:
           None
       """
    active_sectors = input_manager.active_sectors
    active_regions = input_manager.active_regions
    input_data_paths = input_manager.sector_paths
    #extract the list of uselul energy types that are needed to be processed
    ue_type_list = input_manager.general_settings.UE_settings.loc[
        input_manager.general_settings.UE_settings['Parameter'] == "Useful energy types for calculation", 'Value'
    ].iloc[0].split(",")
    ue_type_list = [v.strip() for v in ue_type_list]
    sectors = [] # list of sector instances
    #intilize the Ddr data object
    demand_driver = DemandDriver()
    demand_driver.global_ddr_values = set()
    excel_cache = ExcelFileCache()
    hist_data, user_set_data = load_sector_data( input_data_paths, excel_cache)
    hist_data = hist_data[
        (hist_data['Type'].isna()) | (hist_data['Type'].isin(ue_type_list))
        ]
    user_set_data = user_set_data[
        (user_set_data['Type'].isna()) | (user_set_data['Type'].isin(ue_type_list))
        ]
    for sector_name in active_sectors:
        # Get the Sector object by name
        sector = Sector.get_by_name(sector_name)
        sector_settings = sector.sector_settings
        hist_data_by_sector = hist_data[hist_data['Sector'] == sector_name]
        user_set_data_by_sector = user_set_data[user_set_data['Sector'] == sector_name]
        hist_data_by_sector.columns = hist_data_by_sector.columns.map(str)
        user_set_data_by_sector.columns = user_set_data_by_sector.columns.map(str)
        extract_unique_ddr_values(user_set_data,demand_driver)
        for subsector_name, row in sector_settings.iterrows():
            # Initialize Subsector
            subsector = Subsector(name=subsector_name)
            # Get technologies, ensuring "default" exists if no technologies are provided
            technology_names = (
                [tech.strip() if tech.strip() else "default" for tech in row["Technology"].split(",")]
                if "Technology" in row and pd.notna(row["Technology"]) and row["Technology"].strip()
                else ["default"])
            # Process ECU of the subsector
            ecu_name = row.get("ECU")
            # Process DDet Variables
            ddet_columns = {col: row[col] for col in row.index if col.startswith("DDet") and pd.notna(row[col])}
            ddet_values = list(ddet_columns.values())
            subsector_variables = [ecu_name] + ddet_values # the first element always ecu
            for index, variable_name in enumerate(subsector_variables):
                if index == 0:  # ECU Variable
                    variable = Variable(name=variable_name)
                    # Populate ECU regions
                    populate_variable(variable =variable,
                                               hist_data = hist_data_by_sector,
                                              user_set_data = user_set_data_by_sector,
                                              active_regions = active_regions,
                                              subsector_name = subsector_name)
                    subsector.ecu = variable
                else:  # DDet Variables
                    for tech_name in technology_names:
                        variable = Variable(name=variable_name)

                        # Check if there is any data for the current variable and technology
                        variable_user_df = user_set_data_by_sector[
                            (user_set_data_by_sector["Variable"] == variable.name) &
                            (user_set_data_by_sector["Subsector"] == subsector_name) &
                            (user_set_data_by_sector["Region"].isin(active_regions)) &
                            (user_set_data_by_sector["Technology"] == tech_name)
                            ]
                        if variable_user_df.empty:  # Skip if there is no data for this technology
                            continue
                        # Get or create Technology
                        technology = next(
                            (tech for tech in subsector.technologies if tech.name == tech_name),
                            None)
                        if not technology:
                            technology = Technology(name=tech_name)
                            subsector.technologies.append(technology)

                        # Populate variable for the specific technology
                        populate_variable(variable = variable,
                                                  hist_data = hist_data_by_sector,
                                                  user_set_data = user_set_data_by_sector,
                                                  tech_name = tech_name,
                                                  active_regions = active_regions,
                                                  subsector_name = subsector_name)
                        technology.ddets.append(variable)
                # Add Subsector to Sector
            sector.subsectors.append(subsector)
            #print(sector)
        sectors.append(sector)
    demand_driver.read_all_demand_drivers(input_manager)
    excel_cache.clear_cache()
    return sectors, demand_driver

def populate_variable(variable, hist_data, user_set_data, active_regions,subsector_name, tech_name=None):
    """
    Populate region-specific data for a given variable based on active regions and technology.

    Args:
        variable (Variable): The variable object to populate.
        hist_data (pd.DataFrame): Historical data for the sector.
        user_set_data (pd.DataFrame): User-set data for the sector.
        active_regions (list): List of active regions.
        tech_name (str, optional): Filter by technology name. Defaults to None.
    """
    # Filter user-set data by variable name, active regions, and (optionally) technology
    variable_user_df = user_set_data[(user_set_data["Variable"] == variable.name) & (user_set_data["Subsector"] == subsector_name) &
                                     (user_set_data["Region"].isin(active_regions))]
    # Filter historical data by variable name, active regions, and (optionally) technology
    variable_hist_df = hist_data[(hist_data["Variable"] == variable.name) & (hist_data["Subsector"] == subsector_name) &
                                 (hist_data["Region"].isin(active_regions))]
    if tech_name:
        variable_user_df = variable_user_df[variable_user_df["Technology"] == tech_name]
        variable_hist_df = variable_hist_df[variable_hist_df["Technology"] == tech_name]
    active_regions = [region for region in active_regions if region != "default"]
    regions = load_regions_data_for_variable (variable_user_df = variable_user_df,
                                              variable_hist_df = variable_hist_df,
                                              active_regions = active_regions)
    variable.region_data = []  # Reset before adding new data
    for region in regions:
        variable.region_data.append(region)

def load_regions_data_for_variable(variable_user_df, variable_hist_df, active_regions):
    """
    Load region-specific data for a variable based on user-set and historical data.

    Args:
        variable_user_df (pd.DataFrame): Filtered user-set data for variable.
        variable_hist_df (pd.DataFrame): Filtered historical data for variable.
        active_regions (list): List of active regions to process.
    Returns:
        list: List of Region objects populated with relevant data.
    """
    settings_columns = [
                           col for col in
                           ["Region", "Type", "Temp_level","Subtech","Drive", "Unit", "Factor", "Function", "Equation", "Forecast data", "Lower limit"]
                           if col in variable_user_df.columns
                       ] + [col for col in variable_user_df.columns if str(col).startswith("DDr")]

    filter_columns = ["Type","Temp_level","Subtech","Drive"] #TODO
    regions = []
    # Filter default rows
    default_set_rows = variable_user_df[variable_user_df["Region"] == "default"]
    default_hist_rows = variable_hist_df[variable_hist_df["Region"] == "default"]
    variable_set_df = variable_user_df[settings_columns]

    for region_name in active_regions:
        region = Region(region_name=region_name)  # Initialize Region object
        # Get region-specific settings
        region_set_df = get_region_data(variable_set_df, region_name, default_set_rows, filter_columns)
        region.settings = region_set_df  # Save settings for the region
        filtered_list = [col for col in filter_columns if col in region_set_df.columns]

        if any(col in region_set_df.columns for col in filtered_list):
            # Separate historical and user-set data for all types
            historical_rows = region_set_df[region_set_df["Forecast data"] == "Historical"]
            user_rows = region_set_df[region_set_df["Forecast data"] != "Historical"]
            if not historical_rows.empty:
                filter_keys = [col for col in filtered_list if
                               col in historical_rows.columns and col in variable_hist_df.columns]
                # Extract existing region data based on filter keys
                region_types_levels = historical_rows[filter_keys]

                # Filter user data for the specific region
                hist_data_region = variable_hist_df[
                    (variable_hist_df["Region"] == region_name) &
                    variable_hist_df[filter_keys].apply(tuple, axis=1).isin(region_types_levels.apply(tuple, axis=1))
                    ]
                # Identify missing row combinations (not just Type, but all keys)
                missing_combinations = region_types_levels.apply(tuple, axis=1)[
                    ~region_types_levels.apply(tuple, axis=1).isin(hist_data_region[filter_keys].apply(tuple, axis=1))
                ]

                # Get missing data from the "default" region (based on missing row combinations)
                if not missing_combinations.empty:
                    default_data = variable_hist_df[
                        (variable_hist_df["Region"] == "default") &
                        variable_hist_df[filter_keys].apply(tuple, axis=1).isin(missing_combinations)
                        ]
                    hist_data_region = pd.concat([hist_data_region, default_data], ignore_index=True)

                region.historical = hist_data_region

            if not user_rows.empty:
                filter_keys = [col for col in filtered_list if
                               col in user_rows.columns and col in variable_user_df.columns]
                # Extract existing region data based on filter keys
                region_types_levels = user_rows[filter_keys]

                # Filter user data for the specific region
                user_data_region = variable_user_df[
                    (variable_user_df["Region"] == region_name) &
                    variable_user_df[filter_keys].apply(tuple, axis=1).isin(region_types_levels.apply(tuple, axis=1))
                    ]

                # Identify missing row combinations (not just Type, but all keys)
                missing_combinations = region_types_levels.apply(tuple, axis=1)[
                    ~region_types_levels.apply(tuple, axis=1).isin(user_data_region[filter_keys].apply(tuple, axis=1))
                ]
                # Get missing data from the "default" region (based on missing row combinations)
                if not missing_combinations.empty:
                    default_data = variable_user_df[
                        (variable_user_df["Region"] == "default") &
                        variable_user_df[filter_keys].apply(tuple, axis=1).isin(missing_combinations)
                        ]
                    user_data_region = pd.concat([user_data_region, default_data], ignore_index=True)

                region.user = user_data_region

        else:
            # Handle cases without iteration columns
            forecast_data = region.settings["Forecast data"].iloc[0]
            if forecast_data == "Historical":
                hist_data = get_region_data(variable_hist_df, region_name, default_hist_rows,filtered_list)
                region.historical = hist_data
            else:
                user_data = get_region_data(variable_user_df, region_name, default_set_rows,filtered_list)
                region.user = user_data
        # Add the region to the list if it has relevant data
        if (region.user is not None and not region.user.empty) or (
                region.historical is not None and not region.historical.empty
        ):
            regions.append(region)

    return regions

def get_region_data(df, region_name, default_data, filter_columns):
    # Filter df for the given region_name
    region_data = df[df['Region'] == region_name]
    if region_data.empty:
        return clean_dataframe(default_data)
    filtered_col = [col for col in filter_columns if col in region_data.columns]
    if filtered_col:
        default_data = default_data.reset_index(drop=True)
        region_data = region_data.reset_index(drop=True)
        # identify missing rows
        merged = default_data.merge(region_data, on=filtered_col, how="left", indicator=True)
        # Extract missing rows (rows that exist in default but not in region_settings)
        missing_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        # Append missing rows to region dataset
        # Drop `_x` and `_y` suffixes from duplicate columns
        missing_rows = missing_rows.loc[:, ~missing_rows.columns.str.endswith("_y")]
        missing_rows.columns = missing_rows.columns.str.replace("_x", "", regex=True)
        #Ensure all columns exist before concatenation
        common_cols = list(set(region_data.columns).intersection(set(missing_rows.columns)))
        missing_rows = missing_rows[common_cols]
        for col in region_data.columns:
            if col not in missing_rows.columns:
                missing_rows[col] = None

        if not missing_rows.empty:
            region_data = pd.concat([region_data, missing_rows], ignore_index=True)

    return clean_dataframe(region_data)

def clean_dataframe(df):
    if df is None:
        return df
    # Drop rows/columns with all NaN values
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df

def load_sector_data(data_paths,excel_cache):
    # load data file only once for processed sector and save it in the cash to not open it several times after we clear cash
    hist_path = data_paths["hist_path"]
    user_set_path = data_paths["user_set_path"]
    # Use ExcelFileCache to load files
    hist_data = excel_cache.get_excel_file(hist_path)
    user_set_data = excel_cache.get_excel_file(user_set_path)
    return hist_data, user_set_data

def extract_unique_ddr_values(user_set_data,demand_driver):
    """
        Extract unique DDr values from user_set_data and add them to demand_driver.global_ddr_values
        if they are not already present.
        Args:
            user_set_data (pd.DataFrame): The DataFrame containing user-set data with DDr columns.
            demand_driver (DemandDriver): The DemandDriver object to update with unique DDr values.
     """
    # Extract columns starting with "DDr"
    columns = [col for col in user_set_data.columns if str(col).startswith("DDr")]
    # Extract unique values from these columns
    unique_ddr_names = pd.unique(user_set_data[columns].values.ravel())
    # Filter out NaN values
    unique_ddr_names = [ddr for ddr in unique_ddr_names if pd.notna(ddr)]
    # Add only missing DDr values to the global set
    for ddr in unique_ddr_names:
        if ddr not in demand_driver.global_ddr_values:
            demand_driver.global_ddr_values.add(ddr)

class ExcelFileCache:
    """
       A class for caching and efficiently managing Excel files to not open them several times
       Attributes:
           cache (dict): A dictionary for storing preloaded Excel files.
       """
    def __init__(self):
        self.cache = {}

    def get_excel_file(self, file_path):
        """
        Retrieve an Excel file from the cache. If not cached, load it into memory.
        Args:
            file_path (Path or str): Path to the Excel file.
        Returns:
            pd.DataFrame: The "Data" sheet loaded as a DataFrame.
        """
        file_path = file_path
        if file_path not in self.cache:
            # Cache the Excel file
            self.cache[file_path] = pd.ExcelFile(file_path)

        # Return the "Data" sheet as a DataFrame
        return self.cache[file_path].parse(sheet_name="Data")

    def clear_cache(self):
        """
        Clear the file cache to free up memory.
        """
        self.cache.clear()


