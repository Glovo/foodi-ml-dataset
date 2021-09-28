# Adaptive Cross-modal Embeddings for Image-Text Alignment (ADAPT)

This code implements a novel approach for training image-text alignment models, namely ADAPT.

<p align="center">
    <img src="assets/adapt.png" width="700"/>
</p>
<!-- future link: https://github.com/jwehrmann/retrieval.pytorch/assets/adapt.png -->

ADAPT is designed to adjust an intermediate representation of instances from a modality _a_ using an embedding vector of an instance from modality _b_. Such an adaptation is designed to filter and enhance important information across internal features, allowing for guided vector representations – which resembles the working of attention modules, though far more computationally efficient. For further information, please read our [AAAI 2020 paper](https://www.researchgate.net/publication/337636199_Adaptive_Cross-modal_Embeddings_for_Image-Text_Alignment).



## Table of Contents

* [Installation](#installation)
* [Quick start](#quickstart)
* [Training models](#training)
* [Pre-trained models](#pretrained)
* [Citation](#citation)
* [Poster](#poster)

## Installation
<a name="installation"/>

### 1. Python 3 & Anaconda

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://docs.anaconda.com/anaconda/install/) and then create an environment.

### 2. As standalone project

```
conda create --name adapt python=3
conda activate adapt
git clone https://github.com/jwehrmann/retrieval.pytorch
cd retrieval.pytorch
pip install -r requirements.txt
```

### 3. Download datasets

```
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
```

## Quick start
<a name="quickstart"/>


### Setup

* Option 1:

```
conda activate adapt
export DATA_PATH=/path/to/dataset
```

* Option 2:

You can also create a shell alias (shortcut to reference a command). For example, add this command to your shell profile:
```
alias adapt='source activate adapt && export DATA_PATH=/path/to/dataset' 
```

And then only run the declared name of the alias to have everything configured:
```
$ adapt
```

## Training Models
<a name="training"/>

You can reproduce our main results using the following scripts.

* Training on Flickr30k:
```
python run.py options/adapt/f30k/t2i.yaml
python test.py options/adapt/f30k/t2i.yaml -data_split test
python run.py options/adapt/f30k/i2t.yaml
python test.py options/adapt/f30k/i2t.yaml -data_split test
```

* Training on MS COCO:
```
python run.py options/adapt/coco/t2i.yaml
python test.py options/adapt/coco/t2i.yaml -data_split test
python run.py options/adapt/coco/i2t.yaml
python test.py options/adapt/coco/i2t.yaml -data_split test
```

### Ensembling results

To ensemble multiple models (ADAPT-Ens) one can use: 

* MS COCO models:
```
python test_ens.py options/adapt/coco/t2i.yaml options/adapt/coco/i2t.yaml -data_split test
```

* F30k models:
```
python test_ens.py options/adapt/f30k/t2i.yaml options/adapt/f30k/i2t.yaml -data_split test
```

### Pre-trained models
<a name="pretrained"/>

We make available all the main models generated in this research. Each file has the best model of the run (according to validation result), the last checkpoint generated, all tensorboard logs (loss and recall curves), result files, and configuration options used for training. 

#### F30k models:

| Dataset| Model      | Image Annotation R@1 | Image Retrieval R@1 |
|:--:    | :--:       | :--:                | :--:                 |
| F30k   | [ADAPT-t2i](https://wehrmann.s3-us-west-2.amazonaws.com/adapt_models/f30k_adapt_t2i.tar)  |   76.4%                  |   57.8%                  |
| F30k   | [ADAPT-i2t](https://wehrmann.s3-us-west-2.amazonaws.com/adapt_models/f30k_adapt_i2t.tar)  | 66.3%                   |   53.8%                    |
| F30k | ADAPT-ens | 76.2%    | 60.5%   | 
| COCO | [ADAPT-t2i](https://wehrmann.s3-us-west-2.amazonaws.com/adapt_models/coco_adapt_t2i.tar) | 75.4% |  64.0%    | 
| COCO | [ADAPT-i2t](https://wehrmann.s3-us-west-2.amazonaws.com/adapt_models/coco_adapt_i2t.tar) | 67.2%    | 57.8%   | 
| COCO | ADAPT-ens | 75.3%    | 64.4%   | 

## Citation
<a name="citation"/>

If you find this research or code useful, please consider citing our paper:

```
@article{wehrmanna2020daptive,
  title={Adaptive Cross-modal Embeddings for Image-Text Alignment},
  author={Wehrmann, J{\^o}natas and Kolling, Camila and Barros, Rodrigo C},
  booktitle={The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2020)},
  year={2020}
}
```


## Poster
<a name="poster"/>

<p align="center">
    <img src="assets/adapt_poster.png" width="300"/>
</p>
