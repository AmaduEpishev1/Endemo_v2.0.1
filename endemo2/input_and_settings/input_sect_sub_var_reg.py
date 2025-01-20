import pandas as pd


class RegionData:

    def __init__(self, region_name, settings=None):
        self.region_name = region_name
        self.settings = settings  # Settings DataFrame
        self.historical = {}  # Initialize as dictionary {type: DataFrame}
        self.user = {}  # Initialize as dictionary {type: DataFrame}
        self.types = {}  # {Type: Historical/User DataFrame}
        self.ddr_values = []  # Collect DDr values dynamically that don't repeat to load DDr data once


    def add_type_data(self, typ, data, data_origin):
        """
        Add type-specific data (historical or user).
        """
        if typ not in self.types:
            self.types[typ] = {"historical": None, "user": None}
        self.types[typ][data_origin] = data

    def __str__(self):
        # Display settings rows
        settings_str = (
            f"          Settings Rows: {self.settings.shape[0]}\n{self.settings.to_string(index=False)}"
            if self.settings is not None else "          No Settings"
        )

        # Display historical data
        historical_str = "\n".join(
            f"          Historical Type: {typ}, Rows: {data['historical'].shape[0]}\n{data['historical'].to_string(index=False)}"
            for typ, data in self.types.items() if data.get("historical") is not None
        ) if self.types else "          No Historical Data"

        # Display user data
        user_str = "\n".join(
            f"          User Type: {typ}, Rows: {data['user'].shape[0]}\n{data['user'].to_string(index=False)}"
            for typ, data in self.types.items() if data.get("user") is not None
        ) if self.types else "          No User Data"

        return f"      Region: {self.region_name}\n{settings_str}\n{historical_str}\n{user_str}"

class Technology:
    def __init__(self, name):
        self.name = name
        self.variables = []  # List of DDet Variable objects

    def add_variable(self, variable):
        if variable.var_type != 'DDet':
            raise ValueError("Only DDet variables can be added to Technology")
        self.variables.append(variable)

    def __str__(self):
        if self.variables:
            variables_str = "\n".join(str(var) for var in self.variables)
        else:
            variables_str = "      No DDet Variables"
        return f"    Technology: {self.name}\n{variables_str}"


class Variable:
    def __init__(self, name, var_type='ECU', technology=None):
        """
        Initialize a Variable.

        :param name: Name of the variable.
        :param var_type: Type of the variable ('ECU' or 'DDet').
        :param technology: Associated Technology object if var_type is 'DDet'.
        """
        self.name = name
        self.var_type = var_type  # 'ECU' or 'DDet'
        self.technology = technology  # Technology object if 'DDet'
        self.region_data = []

    def add_region_data(self, region):
        self.region_data.append(region)

    def __str__(self):
        if self.var_type == 'ECU':
            tech_str = ""
        else:
            tech_str = f", Technology: {self.technology.name if self.technology else 'None'}"
        regions_str = "\n".join(str(rd) for rd in self.region_data)
        return f"    Variable: {self.name} (Type: {self.var_type}{tech_str})\n{regions_str}"


class Subsector:
    def __init__(self, name):
        self.name = name
        self.variables = []  # List of Variable objects

    def __str__(self):
        if self.variables:
            variables_str = "\n".join(str(var) for var in self.variables)
        else:
            variables_str = "      No Variables"
        return f"  Subsector: {self.name}\n{variables_str}"

    def add_variable(self, variable):
        self.variables.append(variable)

    def get_variable(self, name):
        return next((var for var in self.variables if var.name == name), None)

    def add_technology(self, technology_name):
        """
        Add a Technology object to the Subsector.
        """
        # Check if the technology already exists
        existing_tech = next((tech for tech in self.technologies if tech.name == technology_name), None)
        if not existing_tech:
            technology = Technology(name=technology_name)
            self.technologies.append(technology)
            return technology
        return existing_tech




class Sector:
    """
    Represents a sector and holds multiple Subsector objects.
    """
    def __init__(self, name):
        self.name = name
        self.subsectors = []  # List of Subsector objects
        self.energy_UE = []   # TODO: holder for UE energy

    def add_subsector(self, subsector):
        self.subsectors.append(subsector)

    def get_subsector(self, name):
        """
        Get a subsector by its name.
        """
        for subsector in self.subsectors:
            if subsector.name == name:
                return subsector
        return None

    def __str__(self):
        subsectors_str = "\n".join(str(subsector) for subsector in self.subsectors)
        return f"Sector: {self.name}\n{subsectors_str}"


class DemandDriver:
    """
    Encapsulates the logic to handle demand driver data for global DDr values.
    """
    def __init__(self, global_ddr_values, input_manager):
        """
        Initialize with global DDr values and InputManager for paths.

        :param global_ddr_values: Set of global DDr values (e.g., {'GDP', 'POP', 'TIME'}).
        :param input_manager: InputManager object to access input paths.
        """
        self.global_ddr_values = global_ddr_values
        self.input_manager = input_manager
        self.demand_drivers = {}  # Dictionary to store loaded demand driver data

        # Load all demand driver data
        self.load_all_demand_drivers()

    def load_all_demand_drivers(self):
        """
        Load data for all global DDr values and store them in the demand_drivers dictionary.
        """
        for ddr in self.global_ddr_values:
            driver_data = DemandDriverData(ddr)

            # Special case for 'TIME' (no external data required)
            if ddr == 'TIME':
                self.demand_drivers[ddr] = driver_data  # Store empty DemandDriverData object
                continue

            # Load data from files
            driver_data.load_data(self.input_manager)
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
                #print(f"Loaded data DDr {self.name}: {self.user}")
            else:
                print(f"User file not found for {self.name}: {user_path}")
        except Exception as e:
            print(f"Error loading data for {self.name}: {e}")

    def get_type(self, data):
        """
        Determine the type of the data (e.g., DataFrame, NoneType).
        :param data: The data to analyze.
        :return: A string representing the type of the data.
        """
        return type(data).__name__

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
        """
        Print an overview of the demand driver data, including data types.
        """
        historical_status = (
            f"Loaded ({self.historical.shape[0]} rows, {self.historical.shape[1]} columns)"
            if isinstance(self.historical, pd.DataFrame) else "Not Found"
        )
        user_status = (
            f"Loaded ({self.user.shape[0]} rows, {self.user.shape[1]} columns)"
            if isinstance(self.user, pd.DataFrame) else "Not Found"
        )

        return (
            f"Demand Driver: {self.name}\n"
            f"  Historical: {historical_status}, Type: {self.get_type(self.historical)}\n"
            f"  User: {user_status}, Type: {self.get_type(self.user)}\n"
        )








