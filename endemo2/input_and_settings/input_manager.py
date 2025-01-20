import os
from pathlib import Path

import pandas as pd
from endemo2.input_and_settings import control_parameters as cp
from endemo2.input_and_settings.control_parameters import ControlParameters,Subsectorsettings

class InputManager:
    """
    The InputManager class connects all types of run data, that is in the form of excel sheets in the
    'input' folder.
    """
    super_path = Path(os.path.abspath(''))
    input_path = super_path / 'input'
    output_path = super_path / 'output'
    general_input_path = input_path / 'general'
    ctrl_file = super_path / 'Set_and_Control.xlsx'

    def __init__(self):
        # Read set and control parameters
        self.ctrl: ControlParameters = InputManager.read_set_and_control_parameters()
        self.general_settings = self.ctrl.GEN_settings
        # Initialize Subsectorsettings
        self.subsector_settings = Subsectorsettings(self.general_settings)
        self.subsector_settings.read_active_sector_settings(self)

        self.active_sectors = self.general_settings.active_sectors
        self.active_regions = self.general_settings.active_regions
        self.sectors_path = self.get_paths_for_active_sectors(self.active_sectors)

    def get_paths_for_active_sectors(self, active_sectors):
        """
        Generate input paths for active sectors.

        Args:
            active_sectors (list): A list of sector names.

        Returns:
            dict: A dictionary with sector names as keys and their paths as values.
        """
        sector_paths = {}
        for sector_name in active_sectors:
            # Build the two types of paths using the class attribute
            hist_path = self.input_path / f"{sector_name}_hist.xlsx"
            user_set_path = self.input_path / f"{sector_name}_user_set.xlsx"

            # Store paths in a dictionary
            sector_paths[sector_name] = {
                "hist_path": hist_path,
                "user_set_path": user_set_path,
            }

        return sector_paths

    @classmethod
    def read_set_and_control_parameters(cls) -> cp.ControlParameters:
        """ Reads Set_and_Control_Parameters.xlsx """
        ctrl_ex = pd.ExcelFile(InputManager.ctrl_file)

        # read control parameters
        GEN_settings = cp.GeneralSettings(pd.read_excel(ctrl_ex, sheet_name="GeneralSet"),
                                          pd.read_excel(ctrl_ex, sheet_name="Regions"),
                                          pd.read_excel(ctrl_ex, sheet_name="Sectors"),
                                          ue_set=pd.read_excel(ctrl_ex, sheet_name="UE_Set"),
                                          ue_fe=pd.read_excel(ctrl_ex, sheet_name="UE->FE")
                                         )
        return  cp.ControlParameters(GEN_settings)

