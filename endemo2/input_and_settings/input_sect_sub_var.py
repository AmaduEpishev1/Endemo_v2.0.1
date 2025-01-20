import pandas as pd


class Region:

    def __init__(self, region_name, settings=None, historical = None, user = None , forecast=None):
        """
         Args:
            region_name (str): Name of the region.
            settings (pd.DataFrame, optional): DataFrame containing settings for the region.
            historical (pd.DataFrame, optional): DataFrame containing historical time series data.
            user (pd.DataFrame, optional): DataFrame containing user-provided coefficients or time series.
            forecast (pd.DataFrame, optional): DataFrame containing forecasted data for the region.
        """
        self.region_name = region_name
        self.settings = settings
        self.historical = historical
        self.user = user
        self.forecast = forecast

    def __str__(self):
        return (
            f"Region: {self.region_name}\n"
            f"  Settings: {self.settings} rows\n"
            f"  Historical Data: {self.historical} rows\n"
            f"  User Data: {self.user} rows\n"
            f"  Forecast Data: {self.forecast} rows"
        )

class Variable:
    def __init__(self, name):
        self.name = name
        self.region_data = []

    def __str__(self):
        regions_str = "\n".join(f"    {str(region)}" for region in self.region_data)
        return f"Variable: {self.name}\n  Regions:\n{regions_str if regions_str else '    No region data'}"


class Subsector:
    def __init__(self, name):
        self.name = name
        self.ecu = None  #  ECU object
        self.technologies = []  # List of Technology objects which will contain DDet variables
        self.energy_UE = [] # List of energy_UE objects for Subsector

    def __str__(self):
        technologies_str = "\n".join(f"    {str(tech)}" for tech in self.technologies)
        return (
            f"Subsector: {self.name}\n"
            f"  ECU: {self.ecu}\n"
            f"  Technologies:\n{technologies_str if technologies_str else '    No technologies'}\n"
            f"  Energy UE: {len(self.energy_UE)} entries"
        )

class Technology:
    def __init__(self, name):
        self.name = name
        self.ddets = []  # List of Variable objects
        self.energy_UE =[] # List of energy_UE objects for Technology

    def __str__(self):
        ddets_str = "\n".join(f"      {str(ddet)}" for ddet in self.ddets)
        return (
            f"Technology: {self.name}\n"
            f"  DDets:\n{ddets_str if ddets_str else '    No DDet variables'}\n"
            f"  Energy UE: {len(self.energy_UE)} entries"
        )


class Sector:
    """
    Represents a sector and holds multiple Subsector objects.
    """
    _registry = {}  # Class-level registry to store objects

    def __init__(self, name):
        self.name = name
        self.sector_settings = None #df that holds active subsectors with the names ECU and DDets
        self.subsectors = []  # List of Subsector objects
        self.energy_UE = None #TODO holder for UE energy for the Sector

        Sector._registry[name] = self  # Register the object by its name

    def __str__(self):
        subsectors_str = "\n".join(f"  {str(subsector)}" for subsector in self.subsectors)
        return (
            f"Sector: {self.name}\n"
            f"  Settings: {len(self.sector_settings) if self.sector_settings is not None else 0} rows\n"
            f"  Subsectors:\n{subsectors_str if subsectors_str else '    No subsectors'}\n"
            f"  Energy UE: {'Available' if self.energy_UE else 'None'}"
        )

    @classmethod
    def get_by_name(cls, name):
        """Retrieve a Sector object by its name."""
        return cls._registry.get(name)


class DemandDriver:
    """
    Encapsulates the logic to handle demand driver data for global DDr values.
    """
    def __init__(self):
        """
        Initialize with global DDr values and InputManager for paths.
        """
        self.global_ddr_values = [] # list of DDr nsme sused in the execution of code
        self.demand_drivers = {}  # Dictionary to store loaded demand driver data

    def read_all_demand_drivers(self,input_manager):
        """
        Load data for all global DDr values and store them in the demand_drivers dictionary.
        """
        for ddr in self.global_ddr_values:
            # Special case for 'TIME' (no external data required)
            if ddr == 'TIME':
                continue
            driver_data = DemandDriverData(ddr)
            driver_data.load_data(input_manager)
            self.demand_drivers[ddr] = driver_data

    def get_driver_data(self, driver_name):
        """
        Retrieve the data for a specific demand driver.

        :param driver_name: Name of the demand driver.
        :return: A DemandDriverData object or None if not found.
        """
        return self.demand_drivers.get(driver_name)

    def __str__(self):
        """
        Print an overview of all demand driver data.
        """
        result = [str(driver) for driver in self.demand_drivers.values()]
        return "\n".join(result)


class DemandDriverData:
    """
    Encapsulates data for a specific demand driver, including Historical and User data.
    """
    def __init__(self, name):
        """
        :param name: Name of the demand driver (e.g., 'POP', 'GDP'...).
        """
        self.name = name
        self.historical = None  # Holds historical data
        self.user = None        # Holds user data

    def load_data(self, input_manager):
        """
        Load Historical and User data for the demand driver.
        :param input_manager: InputManager instance for file path access.
        """
        try:
            historical_path = input_manager.general_input_path / f"{self.name}_Historical.xlsx"
            user_path = input_manager.general_input_path / f"{self.name}_User.xlsx"

            # Load Historical data
            if historical_path.exists():
                self.historical = pd.read_excel(historical_path, sheet_name="Data")
            else:
                print(f"Historical file not found for {self.name}: {historical_path}")

            # Load User data
            if user_path.exists():
                self.user = pd.read_excel(user_path, sheet_name="Data")
            else:
                print(f"User file not found for {self.name}: {user_path}")
        except Exception as e:
            print(f"Error loading data for {self.name}: {e}")

    def get_data_for_region(self, region_name, data_origin):
        """
        Retrieve the row for a specific region from Historical or User data.

        :param region_name: Name of the region to fetch.
        :param data_origin: Type of data to extract ('historical' or 'user').
        :return: DataFrame row(s) for the specified region.
        """
        data = getattr(self, data_origin)
        if data is None:
            print(f"No {data_origin} data available for {self.name}.")
            return None

        # Fetch the row for the specified region
        region_data = data[data["Region"] == region_name]
        return region_data

    def __str__(self):
        return (
            f"DDr name: {self.name}\n"
            f"  historical Data: {self.historical} rows\n"
            f"  User Data: {self.user} rows\n"
        )




