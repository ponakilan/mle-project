import pickle
import sys
import logging
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from loader import load_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


class PreprocessDataset:
    def __init__(
            self,
            gdsc_main_df: pd.DataFrame,
            cell_lines_df: pd.DataFrame,
            compounds_df: pd.DataFrame,
            encoders_save_path: str,
            scalers_save_path: str,
            num_records: int,
    ):
        self.gdsc_main_df = gdsc_main_df
        self.cell_lines_df = cell_lines_df
        self.compounds_df = compounds_df
        self.KEEP_COLUMNS = [
            "LN_IC50",
            "Gene Expression",
            "Copy Number Alterations (CNA)",
            "Methylation",
            "TARGET",
            "TARGET_PATHWAY",
            "DRUG_NAME_x",
            "CELL_LINE_NAME",
            "TCGA_DESC",
            "Cancer Type\n(matching TCGA label)",
            "GDSC\nTissue descriptor 1",
            "GDSC\nTissue\ndescriptor 2",
            "Screen Medium",
            "Growth Properties"
        ]
        self.NORMALIZE_COLUMNS = [
            "Gene Expression",
            "Copy Number Alterations (CNA)",
            "Methylation"
        ]
        self.encoders_save_path = encoders_save_path
        self.scalers_save_path = scalers_save_path
        self.num_records = num_records
        self.gdsc_preprocessed_df = None

    def merge_cell_lines(self, gdsc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add COSMIC details to the GDSC dataframe.
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
        """
        logger.info("Adding drug details to the GDSC dataframe.")
        return pd.merge(
            gdsc_df,
            self.compounds_df,
            how='left',
            left_on='DRUG_ID',
            right_on='DRUG_ID'
        )

    def encode_categorical_features(self, gdsc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the categorical features in the GDSC dataframe.
        """
        encoders = dict()
        logger.info("Encoding categorical features...")
        for col in gdsc_df.columns:
            if gdsc_df[col].dtype == "object":
                encoder = LabelEncoder().fit(gdsc_df[col].astype(str).tolist())
                encoders[col] = encoder
                gdsc_df[col] = encoder.transform(gdsc_df[col].astype(str).tolist())

        logger.info("Finished encoding categorical features.")
        with open(self.encoders_save_path, 'wb') as f:
            pickle.dump(encoders, f)

        logger.info(f"Categorical features saved to '{self.encoders_save_path}'.")
        return gdsc_df

    def normalize_features(self, gdsc_df: pd.DataFrame) -> pd.DataFrame:
        scalers = dict()
        logger.info("Normalizing categorical features")
        for col in self.NORMALIZE_COLUMNS:
            scaler = StandardScaler().fit(gdsc_df[col].values.reshape(-1, 1))
            scalers[col] = scaler
            gdsc_df[col] = scaler.transform(gdsc_df[col].values.reshape(-1, 1))

        with open(self.scalers_save_path, "wb") as f:
            pickle.dump(scalers, f)

        return gdsc_df

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess and combine data from different datasets.
        """
        logger.info("Starting preprocessing...")
        gdsc_df_with_cell_lines = self.merge_cell_lines(self.gdsc_main_df)
        gdsc_df_combined = self.merge_compounds(gdsc_df_with_cell_lines)
        gdsc_df_combined = gdsc_df_combined.loc[:, self.KEEP_COLUMNS]
        gdsc_df_combined = gdsc_df_combined.dropna()
        gdsc_df_combined = self.encode_categorical_features(gdsc_df_combined)
        gdsc_df_combined = self.normalize_features(gdsc_df_combined)

        # Shuffle the dataset and take only the required rows
        gdsc_df_combined = gdsc_df_combined.sample(frac=1).reset_index(drop=True)
        gdsc_df_combined = gdsc_df_combined.iloc[:self.num_records]

        self.gdsc_preprocessed_df = gdsc_df_combined
        logger.info(f"Finished preprocessing. Combined dataset is of shape {gdsc_df_combined.shape}.")
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
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--encoders_path", type=str, required=True)
    parser.add_argument("--scalers_path", type=str, required=True)
    parser.add_argument("--num_records", type=int, required=True)
    args = parser.parse_args()

    gdsc_main_df, cell_lines_df, compounds_df = load_datasets(args.yaml_path)

    preprocessor = PreprocessDataset(
        gdsc_main_df=gdsc_main_df,
        cell_lines_df=cell_lines_df,
        compounds_df=compounds_df,
        encoders_save_path=args.encoders_path,
        scalers_save_path=args.scalers_path,
        num_records=args.num_records
    )
    preprocessor.save(args.file_path)
