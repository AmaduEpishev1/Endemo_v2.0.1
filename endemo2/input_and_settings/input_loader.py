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
    sectors_path  = input_manager.sectors_path
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
    for sector_name in active_sectors:
        # Get the Sector object by name
        sector = Sector.get_by_name(sector_name)
        sector_settings = sector.sector_settings
        hist_data, user_set_data = load_sector_data(sector_name,sectors_path,excel_cache)
        hist_data = hist_data[
            (hist_data['Type'].isna()) | (hist_data['Type'].isin(ue_type_list))
            ]
        user_set_data = user_set_data[
            (user_set_data['Type'].isna()) | (user_set_data['Type'].isin(ue_type_list))
            ]
        hist_data.columns = hist_data.columns.map(str)
        user_set_data.columns = user_set_data.columns.map(str)
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
                variable = Variable(name=variable_name)
                if index == 0:  # ECU Variable
                    # Populate ECU regions
                    populate_variable(variable =variable,
                                               hist_data = hist_data,
                                              user_set_data = user_set_data,
                                              active_regions = active_regions,
                                              subsector_name = subsector_name)
                    subsector.ecu = variable
                else:  # DDet Variables
                    for tech_name in technology_names:
                        # Get or create Technology
                        technology = next(
                            (tech for tech in subsector.technologies if tech.name == tech_name),
                            None)
                        if not technology:
                            technology = Technology(name=tech_name)
                            subsector.technologies.append(technology)
                        # Populate variable for the specific technology
                        populate_variable(variable = variable,
                                                  hist_data = hist_data,
                                                  user_set_data = user_set_data,
                                                  tech_name = tech_name,
                                                  active_regions = active_regions,
                                                  subsector_name = subsector_name)
                        technology.ddets.append(variable)
                # Add Subsector to Sector
            sector.subsectors.append(subsector)
            print(sector)
        sectors.append(sector)
        excel_cache.clear_cache()
    demand_driver.read_all_demand_drivers(input_manager)
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

    Returns:
        None
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
    for region in regions:
        variable.region_data.append(region)

def load_regions_data_for_variable(variable_user_df, variable_hist_df,active_regions):
    """
    Load region-specific data for a variable based on user-set and historical data.

    Args:
        variable_user_df (pd.DataFrame): Filtered user-set data.
        variable_hist_df (pd.DataFrame): Filtered historical data.

    Returns:
        list: List of Region objects populated with relevant data.
    """
    settings_columns = (
            [col for col in ["Region","Type", "Temp_level", "Unit", "Factor", "Function", "Equation", "Forecast data"]
             if col in variable_user_df.columns] +
            [col for col in variable_user_df.columns if str(col).startswith("DDr")]
    )
    iteration_columns = ["Type"] # Drive could be added or any other #TODO to ask Andelka what to do with the new columns if they appear
    regions = []
    variable_set_df = variable_user_df[settings_columns]
    # Cache default rows
    default_set_rows = variable_set_df[variable_set_df["Region"] == "default"]
    default_hist_rows = variable_hist_df[variable_hist_df["Region"] == "default"]
    default_user_rows = variable_user_df[variable_user_df["Region"] == "default"]
    for region_name in active_regions:
        region = Region(region_name = region_name) #intiilization of the region object
        # Get region-specific settings
        region_set_row = get_region_data(variable_set_df, region_name, default_set_rows)
        region.settings = region_set_row  # will contain a df that includes settings_columns and also other types
        df_hist = []
        df_user = []
        if any(col in region_set_row.columns for col in iteration_columns): #TODO itteration makes the duplicates  to double check
            for index, row in region_set_row.iterrows():
                value = row.get("Type") #values_series
                forecast_data = row.get("Forecast data") # here is get as itterows converted this into series
                if forecast_data == "Historical":
                    # Load historical data
                    region_data_df = get_region_data(variable_hist_df, region_name, default_hist_rows, value)
                   # years_hist_col = [col for col in region_data_df.columns if str(col).isdigit()] + ['Type'] #TODO it would be much better if we could filter datat here and have pure datat in the objects
                   # region_data_df = region_data_df[years_hist_col]
                    df_hist.append(region_data_df)
                else:
                    # Load user-set data
                    region_data_df = get_region_data(variable_user_df, region_name, default_user_rows, value)
                   # user_columns = [col for col in region_data_df.columns if col not in settings_columns] + ['Type']
                   # region_data_df = region_data_df[user_columns]
                    df_user.append(region_data_df)
                # Concatenate dataframes, or set to None if no data
            hist_df = pd.concat(df_hist, ignore_index=True) if df_hist else None
            region.historical = hist_df.drop_duplicates() if hist_df is not None else None #TODO this seems to be not right but it works
            user_df = pd.concat(df_user, ignore_index=True) if df_user else None
            region.user = user_df.drop_duplicates() if user_df is not None else None
            regions.append(region)
        else:
            forecast_data = region.settings["Forecast data"].iloc[0]
            if forecast_data == "Historical":
                # Load historical data
                region_data_df = get_region_data(variable_hist_df, region_name, default_hist_rows)
                #years_hist_col = [col for col in region_data_df.columns if str(col).isdigit()]
               # region_data_df = region_data_df[years_hist_col]
                region.historical = region_data_df.drop_duplicates()
            else:
                # Load user-set data
                region_data_df = get_region_data(variable_user_df, region_name, default_user_rows)
                #user_columns = [col for col in region_data_df.columns if col not in settings_columns] + ["Type"]
                #region.user = region_data_df[user_columns]
                region.user = region_data_df.drop_duplicates()
        regions.append(region)

    return regions

def get_region_data(df, region_name, default_df, value=None):
    """
    Retrieve region-specific data from a DataFrame, falling back to default if unavailable.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        region_name (str): Region name to filter for.
        default_df (pd.DataFrame): Default row to return if region data is unavailable.
        value: The value of the iteration for the df (e.g., Type, Drive).
    Returns:
        pd.DataFrame: Cleaned DataFrame with region-specific or default data.
    """
    # Initialize an empty list to collect output rows
    if value == "HEAT": # TODO  this looks wrong
        result_df = []
        region_rows = df[(df["Region"] == region_name) & (df["Type"] == value)]
        default_rows = default_df[default_df["Type"] == value]
        for temp_level in default_rows["Temp_level"].unique():
            # Check if this Temp_level exists in the region_rows
            if temp_level not in region_rows["Temp_level"].values:
                # Append the default row for this missing Temp_level
                missing_row = default_rows[default_rows["Temp_level"] == temp_level]
                result_df.append(missing_row)
            elif temp_level in region_rows["Temp_level"].values:
                region_row = region_rows[region_rows["Temp_level"] == temp_level]
                result_df.append(region_row)
            else:
                return clean_dataframe(region_rows)
        result_df = pd.concat(result_df, ignore_index=True)
        return clean_dataframe(result_df)
    if value is not None:
        region_row = df[(df["Region"] == region_name) & (df["Type"] == value)]
    else:
        region_row = df[df["Region"] == region_name]
    if region_row.empty:
        if value is not None:
            region_row = default_df[default_df["Type"] == value]
        else:
            region_row = default_df

    return clean_dataframe(region_row)

def skip_years_in_df(df: pd.DataFrame, skip_years: [int]):
    for skip_year in skip_years:
        if skip_year in df.columns:
            df.drop(skip_year, axis=1, inplace=True)

def clean_dataframe(df):
    if df is None:
        return df
    # Drop rows/columns with all NaN values
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df

def load_sector_data(sector_name,sectors_path,excel_cache):
    # load data file only once for processed sector and save it in the cash to not open it several times after we clear cash
    sector_paths = sectors_path[sector_name] #dictionary with hist and user paths for the processing sector
    hist_path = sector_paths["hist_path"]
    user_set_path = sector_paths["user_set_path"]
    # Use ExcelFileCache to load files
    hist_data = excel_cache.get_excel_file(hist_path)
    user_set_data = excel_cache.get_excel_file(user_set_path)
    return hist_data,user_set_data

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