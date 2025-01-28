"""
This module contains the in-model representation of all settings found in Set_and_Control_Parameters.xlsx
"""
from __future__ import annotations
import pandas as pd
from endemo2.input_and_settings.input_sect_sub_var import Sector

class ControlParameters:
    """
    The ControlParameter class holds the data given by the Set_and_Control_Parameters.xlsx file.
    It is split in general settings, and settings for each sector, indicating non-overlapping parameters for our model.

    :ivar GeneralSettings general_settings: The settings contained in the "GeneralSettings"-sheet.
    :ivar IndustrySettings industry_settings: The settings contained in the "IND_*"-sheets.
    """

    def __init__(self, GEN_settings: GeneralSettings,):
        self.GEN_settings = GEN_settings

class GeneralSettings:
    """
        The GeneralSettings contain the parameters for the model given in Set_and_Control_Parameters.xlsx in the
        GeneralSettings and Countries sheets, as well as the additional UE_Set and UE->FE sheets.

        :param pd.DataFrame ex_general: The dataframe of the "GeneralSettings"-sheet in Set_and_Control_Parameters.xlsx
        :param pd.DataFrame ex_region: The dataframe of the "Regions"-sheet in Set_and_Control_Parameters.xlsx
        :param pd.DataFrame ex_sectors: The dataframe of the "Sectors"-sheet in Set_and_Control_Parameters.xlsx
        :param pd.DataFrame ue_set: The dataframe of the "UE_Set"-sheet in Set_and_Control_Parameters.xlsx
        :param pd.DataFrame ue_fe: The dataframe of the "UE->FE"-sheet in Set_and_Control_Parameters.xlsx
        """

    def __init__(self, ex_general: pd.DataFrame, ex_region: pd.DataFrame, ex_sectors: pd.DataFrame,
                 ue_set: pd.DataFrame, ue_fe: pd.DataFrame):

        self.active_sectors = ex_sectors[ex_sectors['Value'] == True]['Sector'].tolist() # it is the list of active sectors
        self._parameter_values = ex_general # contain a df of parameters Sheet name General_set
        self.active_regions = ex_region[ex_region['Active'] == True]['Region'].tolist()
        self.subsector_settings = Subsectorsettings(self)

        # Extract forecast-related parameters
        self.forecast_year_start = self._get_forecast_parameter("Forecast year start")
        self.forecast_year_end = self._get_forecast_parameter("Forecast year end")
        self.forecast_year_step = self._get_forecast_parameter("Forecast year step")
        self.UE_settings = ue_set
        self.FE_settings = ue_fe

    def __str__(self):
        return (
            f"Active Sectors: {self.active_sectors}\n"
            f"Active Regions: {self.active_regions}\n"
            f"Parameter Values: {self._parameter_values}\n"
            f"UE Settings: \n{self.UE_settings}\n"
            f"FE Settings: \n{self.FE_settings}\n"
        )

    def _filter_by_active_sectors(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a DataFrame to include only rows where the 'Sector' column matches an active sector.

        :param dataframe: The DataFrame to filter.
        :return: Filtered DataFrame containing only active sectors.
        """
        if 'Sector' not in dataframe.columns:
            raise ValueError("The DataFrame does not contain a 'Sector' column.")

        # Filter rows where the 'Sector' column matches active sectors
        return dataframe[dataframe['Sector'].isin(self.active_sectors)]

    def _get_forecast_parameter(self, parameter_name: str) -> int:
        """
        Retrieve a forecast-related parameter by name.

        :param parameter_name: Name of the parameter in the settings file.
        :return: Integer value of the parameter.
        """
        try:
            # Fill missing values with a placeholder (e.g., -1) or handle appropriately
            self._parameter_values['Value'] = self._parameter_values['Value'].fillna(-1).astype(float).astype(int)
            # Retrieve the value for the given parameter name
            value = self._parameter_values.loc[self._parameter_values['Parameter'] == parameter_name, 'Value']
            if value.empty or value.values[0] == -1:
                raise ValueError(f"Parameter '{parameter_name}' is missing or invalid in the settings.")
            return value.values[0]

        except (KeyError, IndexError, ValueError) as e:
            print(f"Error accessing parameter '{parameter_name}': {e}")
            raise ValueError(f"Parameter '{parameter_name}' is missing or invalid in the settings.")

    def get_forecast_year_range(self) -> range:
        """
        Retrieve the range of years for the forecast.

        :return: A range object representing the forecast years.
        """
        return range(self.forecast_year_start, self.forecast_year_end + 1, self.forecast_year_step)


class Subsectorsettings:
    """
    The Subsectorsettings class is designed as an object that contains a dictionary of DataFrames,
    where each key is the name of a sector (e.g., IND, HOU, TRA, CTS), and the value is the corresponding
    DataFrame for that sector's active subsectors.
    example of acessing; # Access data for a specific sector
    ind_data = subsector_settings.get_subsector_data("IND")
    """
    def __init__(self, general_settings: GeneralSettings):
        self.general_settings = general_settings
        self.subsector_settings = {}

    def __str__(self):
        result = []
        for sector, data in self.subsector_settings.items():
            result.append(f"Sector: {sector}, Data:\n{data}")
        return "\n".join(result)

    def read_active_sector_settings(self, InputManager):
        """
        Reads the sheets corresponding to active sector subsectors from ctrl_file.
        The sheet names follow the format '<sector>_subsectors'.
        creates the Sector objects with their corresponding names and register them, also adds the df of active subsector settings
        """
        try:
            # Load the control Excel file
            ctrl_ex = pd.ExcelFile(InputManager.ctrl_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Control file not found: {InputManager.ctrl_file}. Error: {e}")
        except Exception as e:
            raise Exception(f"Error loading control file {InputManager.ctrl_file}: {e}")
        # Dictionary to hold all sector objects
        # Loop through active sectors
        for sector_name in self.general_settings.active_sectors:
            sheet_name = f"{sector_name}_subsectors"
            if sheet_name in ctrl_ex.sheet_names:
                try:
                    # Read the sheet and store only rows where 'Active' is True
                    df = pd.read_excel(ctrl_ex, sheet_name=sheet_name)
                    df = df[df['Active'] == True] # this ensures that we are processing only active subsecors
                    df = df.set_index('Subsector')
                    sector = Sector(sector_name)
                    sector.sector_settings = df # #TODO now we need to use the settings form the instance
                    self.subsector_settings[sheet_name] = df
                except Exception as e:
                    print(f"Error reading sheet {sheet_name}: {e}")
            else:
                print(f"Sheet {sheet_name} not found in control file. No data for {sector_name}.")
