import os
import numpy as np
import pandas as pd
import shutil
import argparse

"""
This script creates the parquet file that will be consumed by our 
DataLoaders (used in WIT and CLIP).
Additionally, it preprocesses the different text fields (
removing NaN descriptions) and creates the final caption that is used for 
our algorithm (product_name + collection_section + product_description).
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="/mnt/data/DATASET_NAME/",
                        help="Folder where the CSV and the images were "
                             "downloaded.")
    args = parser.parse_args()
    PATH_DATA = args.dataset_path
    PATH_PARQUET = os.path.join(PATH_DATA, 'samples')
    DATASET_CSV = 'DATAFRAME_NAME.csv'

    # READ CSV
    print("Reading CSV")
    samples = pd.read_csv(os.path.join(PATH_DATA, DATASET_CSV))
    print("CSV Read successfully")

    samples = samples.drop_duplicates(
        subset=["product_name", "collection_section", "product_description",
                "hash"])

    samples["product_description"].fillna("", inplace=True)
    samples["sentence"] = \
        np.where(samples["product_name"], samples["product_name"].astype(str),
                 "") + " " + \
        np.where(samples["collection_section"],
                 samples["collection_section"].astype(str), "") + " " + \
        np.where(samples["product_description"],
                 samples["product_description"].astype(str), "")

    samples["sentence"] = samples["sentence"].str.lower()
    samples.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)

    # Keep only necessary columns for modelling
    final_samples = samples[["sentence", "s3_path", "subset", "country_code",
                             "hash"]].reset_index().copy()
    final_samples.rename(columns={"sentence": "caption"}, inplace=True)
    final_samples.rename(columns={"subset": "split"}, inplace=True)
    final_samples.rename(columns={"index": "img_id"}, inplace=True)

    # Remove previous version
    if os.path.exists(PATH_PARQUET):
        shutil.rmtree(PATH_PARQUET)

    # Save as partitioned parquet
    final_samples.to_parquet(
        path=PATH_PARQUET,
        engine="pyarrow",
        index=False,
        partition_cols=["split"],
    )