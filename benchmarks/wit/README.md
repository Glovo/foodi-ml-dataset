# WIT Retrieval benchmark

In this README we give a detailed list of the steps require to replicate the results shown in our paper. 

## 1. System Requiremens

Our experiments run in `python3`. To install the required dependencies, we recommend creating an environment:

1. ```python3 -m venv foodi-ml-venv```


2. ```pip install -r benchmarks/wit/requirements.txt```


3. ```export PYTHONPATH=$PYTHONPATH:<ENTER_PATH_TO_REPO_FOLDER>```

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

## 3. Training the network

To train the WIT network in the Foodi-ML-dataset, please run:

```python benchmarks/wit/train_network.py --dataset-path <ENTER_PATH_TO_DATASET_FOLDER> --epochs 50 --batch-size 160```

This will save the weights of the network that will be later used for evaluation.

## 4. Evaluating WIT

In order to evaluate WIT over the Foodi-ML-Dataset, please run this script pointing to the best network's weights.

```python benchmarks/wit/evaluate_network_bigdata.py --dataset-path <PATH_TO_DATASET_FOLDER> --code-path <PATH_TO_REPO_FOLDER> --model-weights <PATH_TO_MODEL_WEIGHTS>```