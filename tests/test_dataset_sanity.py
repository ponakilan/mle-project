import yaml
import warnings
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Open the dataset paths file
with open("data.yaml", "r") as data_paths_file:
    data_paths = yaml.load(data_paths_file, Loader=yaml.FullLoader)

# Load the datasets
gdsc_main_df = pd.read_excel(data_paths["gdsc_main"])
cell_lines_df = pd.read_excel(data_paths["cell_lines"])
compounds_df = pd.read_csv(data_paths["compounds"])

# Check if the details of all the COSMIC_ID in gdsc_main is present in cell_lines
def test_cosmic_details_availability():
    unique_cosmic_ids = set(gdsc_main_df["COSMIC_ID"])
    unique_cosmic_details = set(cell_lines_df["COSMIC identifier"])
    assert len(unique_cosmic_ids.intersection(unique_cosmic_details)) == len(unique_cosmic_ids)

# Check if the details of all the DRUG_ID in gdsc_main is present in compounds
def test_drug_details_availability():
    unique_drug_ids = set(gdsc_main_df["DRUG_ID"])
    unique_drug_details = set(compounds_df["DRUG_ID"])
    assert len(unique_drug_ids.intersection(unique_drug_details)) == len(unique_drug_ids)
