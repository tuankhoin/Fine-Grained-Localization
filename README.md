# COMP90086 Project - Fine-grained localisation
Team member: Tuan Khoi Nguyen (1025294) and Hoang Anh Huy Luu (1025379)

## Introduction
* The submission contains notebooks, scripts and result files used in the Project. This approach have helped us, a team of 2 fresh Masters students at the time, reaching the 4th position on the [Kaggle leaderboard](https://www.kaggle.com/c/comp90086-2021), consisting of 215 teams in different levels of Graduates, many of them in their final year with much more experience.
* This README file consists of file description with instructions to run scripts, in expected procedual order.

## Data Processing
Note: All data folders are assumed to be put in the same directory as the scripts.

### Feature Extraction
* `CNN_Extraction.ipynb`: This script runs CNN model to extract features, and calculate similarity and return top 3 most similar's average location. Default model is `keras.applications.resnet.ResNet101`. To change model, change the function name in `base_model` to another model. Examples include `ResNet50`, `ResNet101V2`, `ResNet152`,...
* `CNN_Postprocess.ipynb`: This script reads in the CNN extracted features, caclculate similarity and rank by most similar for each test instances. This script export 2 `.csv` files: The sorted filenames and their corresponding similarity metric (Euclidean Distance).
* `SIFT_Extraction.ipynb`: This script get SIFT matches between test instance and train instances, and also rank them on similarity like CNN extraction script (by number of matches). The process will export variables to 2 folders `kp_test` and `kp_train` rather than storing them in RAM memory (the process will take approximately 7GB of space and approximately 44 hours), then process to 2 `.csv` files with the same structure as CNN extracted files. The train process can run on smaller data portions, as the script will only take in instances within `train` and `test` folder.
* `color_extraction.ipynb`: This script works the same as above extraction scripts, with the similarity criteria being number of histogram intersections.

All the extraction results for the above scripts can be found in the Google Drive link below.

### Validation
* `validate_creation.ipynb`: Extract a validation set from train dataset, and preform CNN feature extraction on it and do validation.
* `classify_validation_model.ipynb`: Run adversarial validation to measure representation of the validation dataset.
* `validation_model.ipynb`: Do validation for all ensemble models mentioned in the report.

Our created validation dataset are available in the Google Drive link below.

### Tuning
* `model_tuning.ipynb`: Run 2-Layer Grid Search to get the most optimal CLOCK set of hyperparameters

## Models
* `Random_Centre.ipynb`: Baseline models - Center Allocation and Random Allocation.
* `kmeans.py`: Function for CLOCK and SHOCK clustering models.
* `kmeans_CLOCK.ipynb`: Basic CLOCK implementation.
* `clustering_models.ipynb`: Running Combination model, with CLOCK and SHOCK implemented.
* `cluster_translation.ipynb`: Do translation geometry to retrieve exact location.

### Evaluation
* `experiment_script.ipynb`: Show best match results for each method of receiving similar images.

## Data
Pre-processed data can be downloaded [in this Google Drive URL](https://drive.google.com/drive/folders/1wCfVut7QrmFHKH4pAHWx1acBSLloNZqw?usp=sharing) (UniMelb email access only).

## Results
All results are in `.csv` files, with their names corresponding to the model run.
