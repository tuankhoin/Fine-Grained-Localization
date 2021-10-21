# Submission

This submission includes the files used in COMP90086 project.

The following are file description with instructions to run, in expected procedual order:

## Data Processing
This includes the CNN_extraction file, CNN_Postprocess file the SIFT_extraction file and the color_extraction file. 
To run these 3 files, simply just execute all the cells in the notebooks.
Note: teh CNN_Postprocess needs to be run after the CNN_extraction
### Validation
Firstly, we need to run the validate_resnet101 to create the validation dataset. 
Then, we run the classify_validation_model for Joint Adversarial Validation.
### Tuning
We then run the model_tuning script to find the best hyper parameters.
## Models
Run the Random_Centre for the random and centre points models.
We run the kmeans_CLOCK notebook for all the CLOCK models.
Run the clustering_model followed by cluster_extraction for SHOCK models.

Pre-processed data can be found here (link drive)
https://drive.google.com/drive/folders/1wCfVut7QrmFHKH4pAHWx1acBSLloNZqw?usp=sharing