# Retrieval benchmarks

In this README we give a detailed list of the steps require to replicate the results shown in our paper. 

## 1. System Requiremens

Our experiments run in `python3`. To install the required dependencies, we recommend creating an environment:

```python3 -m venv DATASET_NAME-venv```

```pip install -r benchmarks/wit/requirements.txt```

```export PYTHONPATH=$PYTHONPATH:<ENTER_PATH_TO_REPO_FOLDER>```
## 2. Preprocessing of the dataset

In order to replicate our results, the dataset must be preprocessed first in the same way we did. Running the following script will:

```python benchmarks/benchmark_preprocess.py --dataset-path <ENTER_PATH_TO_DATASET_FOLDER>```

* Generate a parquet file in the folder where you have downloaded the dataset with the following splits:
  * train
  * val
  * test
  
  Each of them containing the columns:
    * caption
    * split: (train, val, test)
    * img_id
    * s3_path: Path to the images
    * country_code
    * hash
    
* Preprocess the different text fields that compose our dataset and aggregate them to one single column called caption. Please note that this is an algorithm based decision and researchers can choose freely how they use the different text fields of our dataset.

## 3. Evaluating CLIP
In order to evaluate CLIP on the DATASET_NAME-dataset, please run the notebook `CLIP-MultiLingual-DATASET_NAME.ipynb`.
