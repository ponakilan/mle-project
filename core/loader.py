import yaml
import warnings
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def load_datasets(yaml_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the data using the paths from the yaml file.
    """
    with open(yaml_file_path, "r") as data_paths_file:
        data_paths = yaml.load(data_paths_file, Loader=yaml.FullLoader)

    gdsc_main_df = pd.read_excel(data_paths["gdsc_main"])
    cell_lines_df = pd.read_excel(data_paths["cell_lines"])
    compounds_df = pd.read_csv(data_paths["compounds"])

    return gdsc_main_df, cell_lines_df, compounds_df
