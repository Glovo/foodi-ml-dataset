# Adaptive Cross-modal Embeddings for Image-Text Alignment (ADAPT) - FooDI-ML dataset adaptation

This code implements a novel approach for training image-text alignment models, namely ADAPT. Forked from the original code here: https://github.com/jwehrmann/retrieval.pytorch.


The code included here includes a mod to be able to run ADAPT for very large image datasets. Essentially, we move some logic in the evaluation scripts so that the matrices supported by GPU training are less intensive.

<!-- future link: https://github.com/jwehrmann/retrieval.pytorch/assets/adapt.png -->

ADAPT is designed to adjust an intermediate representation of instances from a modality _a_ using an embedding vector of an instance from modality _b_. Such an adaptation is designed to filter and enhance important information across internal features, allowing for guided vector representations – which resembles the working of attention modules, though far more computationally efficient. For further information, please read ADAPT's original paper [AAAI 2020 paper](https://www.researchgate.net/publication/337636199_Adaptive_Cross-modal_Embeddings_for_Image-Text_Alignment).



## Table of Contents

* [Installation](#installation)
* [Sample dataset](#sample)
* [Training models](#training)
* [Pre-trained models](#pretrained)
* [Citation](#citation)
* [Poster](#poster)

## Installation
<a name="installation"/>

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://docs.anaconda.com/anaconda/install/) and then create an environment. The benchmark was generated with a GPU, so the installation assumes that a GPU is available.

To install the necessary dependencies the notebook `notebooks/0_install_requirements.ipynb` must be run.

## Sample dataset
<a name="sample"/>

To run the code to generate the metrics on the sample dataset of 10k samples, please follow the next instructions: 

* Download the sample dataframe from S3 by running: `aws s3 cp s3://glovo-products-dataset-d1c9720d/mock_dataset_ES.csv . --no-sign-request`
* Rename the dataframe to `glovo-foodi-ml-dataset.csv`
* Download the sample images (2 GB) from S3 by running: `aws s3 cp s3://glovo-products-dataset-d1c9720d/mock_dataset_ES.zip . --no-sign-request` and unzip the folder. Rename it from `mock_dataset_ES --> dataset`.
* Run the notebook `notebooks/1_preprocess_sample.ipynb` to generate the parquet file that will be consumed by the data loader.

To train the model and generate the benchmark results, run:
```{bash}
cd ~/foodi-ml-dataset/benchmarks/adapt/
source activate python3
export DATA_PATH=PATH_WHERE_DATASET_IS_STORED (the images)
python run.py options/adapt/foodi-ml/i2t.yaml
```

The generated metrics are reported in the validation set because no hyperparameter tuning was performed.


## Citation
<a name="citation"/>

This is a modification from the original work ADAPT.
```
@article{wehrmanna2020daptive,
  title={Adaptive Cross-modal Embeddings for Image-Text Alignment},
  author={Wehrmann, J{\^o}natas and Kolling, Camila and Barros, Rodrigo C},
  booktitle={The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2020)},
  year={2020}
}
```

