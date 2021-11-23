import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="/mnt/data/foodi-ml/", help="Folder where the CSV and the images were downloaded.")
    args = parser.parse_args()
    PATH_DATA = args.dataset_path
    DATASET_CSV = 'glovo-foodi-ml-dataset.csv'

    # READ CSV
    print("Reading CSV...")
    samples = pd.read_csv(os.path.join(PATH_DATA, DATASET_CSV))
    print("CSV read successfully!")

    # Check that images in the dataframe are refering to the correct path
    assert os.path.exists(samples["s3_path"].iloc[0]), f"Image {samples['s3_path'].iloc[0]} not found. Check if the dataframe has the correct image names (see 3rd step of section 1.2 of the README.md)."

    # Some image names (~30) are present in the dataframe but not in the S3 bucket, we remove them.
    missing_images = []
    for p in samples["s3_path"].unique():
        if not os.path.exists(p):
            print(f"Removing {p}")
            missing_images.append(p)

    # Remove them from the dataframe
    samples = samples[~samples["s3_path"].isin(missing_images)]
    # Save the dataframe
    samples.to_csv(os.path.join(PATH_DATA, DATASET_CSV))
    print(f"Dataframe saved at {os.path.join(PATH_DATA, DATASET_CSV)}")
