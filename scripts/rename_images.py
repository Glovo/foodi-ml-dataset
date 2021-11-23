import argparse
import os

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename dataframe column to contain the image names in your disk location.")
    parser.add_argument("--output-dir", type=str, required=True, help="output_dir must be ENTER_DESTINATION_PATH")
    args = parser.parse_args()
    output_dir = args.output_dir # output_dir is ENTER_DESTINATION_PATH
    df = pd.read_csv(os.path.join(output_dir, "glovo-foodi-ml-dataset.csv"))
    print("Changing image names...")
    # df["s3_path"] contains the s3 paths in the format dataset/image_name.png
    df["s3_path"] = df["s3_path"].apply(lambda x: os.path.join(output_dir, x))
    print("Names changed successfully, saving dataframe")
    df.to_csv(os.path.join(output_dir, "glovo-foodi-ml-dataset.csv"))
    print("DataFrame saved successfully")