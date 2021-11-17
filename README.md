<img src="/imgs/grid_img.png" align="top" width="1028" height="234"/>

# DATASET_NAME

This is the GitHub repository for the DATASET_NAME dataset.
This dataset contains over 1.5M unique images and over 9.5M store names, product names, descriptions and collection sections gathered from the APP_NAME application. 
The data made available corresponds to food, drinks and groceries products from over 37 countries in Europe, the Middle East, Africa and Latin America. 
The dataset comprehends 33 languages, including 870k samples of languages of countries from Eastern Europe and West Asia such as Ukrainian and Kazakh, which have been so far underrepresented in publicly available visio-linguistic datasets. 
The dataset also includes widely spoken languages such as Spanish and English.

## License

The DATASET_NAME dataset is offered under the [BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/2.0/ "BY-NC-SA license").

# 1. Download the dataset
The DATASET_NAME dataset is hosted in a S3 bucket in AWS. Therefore AWS CLI is needed to download it. 
Our dataset is composed of:
* One DataFrame (`DATAFRAME_NAME`) stored as a `csv` file containing all text information + image paths in S3. The **size of this CSV file is 540 MB**.
* Set of images listed in the DataFrame. The disk space **required to store all images is 316.1 GB**.

## 1.1. Download AWS CLI
If you do not have AWS CLI already installed, please download the latest version of [AWS CLI](https://aws.amazon.com/cli/ "AWS CLI page") for your operating system.

## 1.2. Download DATASET_NAME
1. Run the following command to download the DataFrame in `ENTER_DESTINATION_PATH` directory. We provide an example as if we were going to download the dataset in the directory `/mnt/data/DATASET_NAME/`.
                                                       
   `aws s3 cp s3://BUCKET_NAME/DATAFRAME_NAME.csv ENTER_DESTINATION_PATH --no-sign-request`

   _Example:_ `aws s3 cp s3://BUCKET_NAME/DATAFRAME_NAME.csv /mnt/data/DATASET_NAME/ --no-sign-request` 

2. Run the following command to download the images in `ENTER_DESTINATION_PATH/dataset` directory (**please note the appending of /dataset**). This command will download the images in `ENTER_DESTINATION_PATH`directory.
 
   `aws s3 cp --recursive s3://BUCKET_NAME/dataset ENTER_DESTINATION_PATH/dataset --no-sign-request --quiet`
           
   _Example:_ `aws s3 cp --recursive s3://BUCKET_NAME/dataset /mnt/data/DATASET_NAME/dataset --no-sign-request --quiet`

3. Run the script `rename_images.py`. This script modifies the DataFrame column to include the paths of the images in the location you specified with `ENTER_DESTINATION_PATH/dataset`.
   ```
   pip install pandas
   python scripts/rename_images.py --output-dir ENTER_DESTINATION_PATH
   ```

# Getting started
Our dataset is managed by the DataFrame `DATAFRAME_NAME`. This dataset contains the following columns:

* **country_code**: This column comprehends 37 unique country codes as explained in our paper. These codes are:

  ```'ES', 'PL', 'CI', 'PT', 'MA', 'IT', 'AR', 'BG', 'KZ', 'BR', 'ME', 'TR', 'PE', 'SI', 'GE', 'EG', 'RS', 'RO', 'HR', 'UA', 'DO', 'KG', 'CR', 'UY', 'EC', 'HN', 'GH', 'KE', 'GT', 'CL', 'FR', 'BA', 'PA', 'UG', 'MD', 'NG', 'PR'```
  
* **city_code**: Name of the city where the store is located.
* **store_name**: Name of the store selling that product. If `store_name` is equal to `AS_XYZ`, it represents an auxiliary store. This means that while the samples contained are for the most part valid, the store name can't be used in learning tasks
* **product_name**: Name of the product. All products have `product_name`, so this column does not contain any `NaN` value.
* **collection_section**: Name of the section of the product, used for organizing the store menu. Common values are _"drinks", "our pizzas", "desserts"_. All products have `collection_section` associated to it, so this column does not have any `NaN` value in it.
* **product_description**: A detailed description of the product, describing ingredients and components of it. **Not all products of our data have description, so this column contains `NaN` values that must be removed by the researchers as a preprocessing step.**
* **subset**: Categorical variable indicating if the sample belongs to the Training, Validation or Test set. The respective values in the DataFrame are `["train", "val", "test"]`. 
* **HIER**: Boolean variable indicating if the store name can be used to retrieve product information (indicating if the store_name is **not** an auxiliary store (with code `AS_XYZ`)).
* **s3_path**: Path of the image of the product in the disk location you chose. 

# Dataset Statistics
A notebook analyzing several dataset statistics is provided in `notebooks/DATASET_NAME Dataset Stats Analytics.ipynb`.

# Benchmark
Our paper includes 3 benchmarks: 

**Text to Image/Image to Text Retrieval**
* [WIT](benchmarks/wit/README.md)
* [CLIP](benchmarks/clip/CLIP-MultiLingual-DATASET_NAME.ipynb)

**Conditional Image Generation**
* SAGAN