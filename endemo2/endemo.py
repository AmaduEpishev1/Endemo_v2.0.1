from endemo2.input_and_settings.input_manager import InputManager
from endemo2.input_and_settings.input_loader import  initialize_hierarchy_input_from_settings_data
from endemo2.model_instance.model_forecast import Forecast
from endemo2.model_instance.model_useful_energy import UsefulEnergyCalculator

class Endemo:
    """
    This is the whole program. From here we control what the model does on the highest level.

    :ivar Input input_manager: holds all the processed input_and_settings from the Excel sheets for the current run of
        the program.
    :ivar dict[str, Country] countries_in_group: holds all the country objects, accessible by the countries_in_group english name.
    :ivar Preprocessor preprocessor: holds an additional layer of preprocessed data, building upon the input_manager.
    """

    def __init__(self):
        self.input_manager = None
        self.input = None
        self.forecast = None
        self.output_manager = None  #TODO

    def execute_with_preprocessing(self):
        """
        Executes the whole program from start to end.
        """
        # Initialize Input
        print("Reading settings...")
        self.input_manager = InputManager()
        print("Settings successfully read.")

        # Read input data for active_regions in active sectors
        print("Reading input data for active sectors...")
        self.input, demand_driver = initialize_hierarchy_input_from_settings_data(self.input_manager)
        print("Input data successfully read.")


        # #Calculate the forecast
        print("Do Predictions...")
        self.forecast = Forecast(self.input, demand_driver, self.input_manager)
        print("Predictions successfully done")


        #Calculate useful energy
        print("Calculate useful energy ...")
        useful_energy = UsefulEnergyCalculator(self.input_manager, self.input)
        print("Calculate useful energy successfully done...")

        # # generate output files #TODO move the outputs to the separate script
        # self.write_all_output()



