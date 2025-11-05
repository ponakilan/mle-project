import sys
import yaml
import logging
import argparse
import warnings
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

class PreprocessDataset:
    def __init__(
            self,
            gdsc_main_df: pd.DataFrame,
            cell_lines_df: pd.DataFrame,
            compounds_df: pd.DataFrame
    ):
        self.gdsc_main_df = gdsc_main_df
        self.cell_lines_df = cell_lines_df
        self.compounds_df = compounds_df
        self.gdsc_preprocessed_df = None

    def merge_cell_lines(self, gdsc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add COSMIC details to the GDSC dataframe.
        :param gdsc_df: GDSC dataframe
        :return: pd.DataFrame
        """
        logger.info("Adding COSMIC details to the GDSC dataframe.")
        return pd.merge(
            gdsc_df,
            self.cell_lines_df,
            how='left',
            left_on='COSMIC_ID',
            right_on='COSMIC identifier'
        )

    def merge_compounds(self, gdsc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add drug details to the GDSC dataframe.
        :param gdsc_df: pd.DataFrame
        :return: pd.DataFrame
        """
        logger.info("Adding drug details to the GDSC dataframe.")
        return pd.merge(
            gdsc_df,
            self.compounds_df,
            how='left',
            left_on='DRUG_ID',
            right_on='DRUG_ID'
        )

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess and combine data from different datasets.
        :return: pd.DataFrame
        """
        logger.info("Starting preprocessing...")
        gdsc_df_with_cell_lines = self.merge_cell_lines(self.gdsc_main_df)
        gdsc_df_combined = self.merge_compounds(gdsc_df_with_cell_lines)
        self.gdsc_preprocessed_df = gdsc_df_combined
        logger.info(f"Finished preprocessing. Combined dataset with {self.gdsc_preprocessed_df.shape[1]} features.")
        return gdsc_df_combined

    def save(self, file_path: str):
        if self.gdsc_preprocessed_df is None:
            logger.info("Data not preprocessed. Initiating preprocessing...")
            self.gdsc_preprocessed_df = self.preprocess()
        self.gdsc_preprocessed_df.to_csv(file_path, index=False)
        logger.info(f"Dataset saved to '{file_path}'")


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Preprocess and combine GDSC details from different datasets."
    )
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()

    # Open the dataset paths file
    with open("data.yaml", "r") as data_paths_file:
        data_paths = yaml.load(data_paths_file, Loader=yaml.FullLoader)

    # Load the datasets
    logger.info("Loading data...")
    gdsc_main_df = pd.read_excel(data_paths["gdsc_main"])
    cell_lines_df = pd.read_excel(data_paths["cell_lines"])
    compounds_df = pd.read_csv(data_paths["compounds"])

    # Preprocess and save the dataset
    preprocessor = PreprocessDataset(
        gdsc_main_df=gdsc_main_df,
        cell_lines_df=cell_lines_df,
        compounds_df=compounds_df
    )
    preprocessor.save(args.filepath)
