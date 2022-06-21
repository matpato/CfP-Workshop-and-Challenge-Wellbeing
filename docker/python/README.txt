# DATASET

The dataset file - dataset_smile_challenge.npy -
must be in the directory "./app" for this to run nicely.

# SCRIPTS

To run the scripts use the command :
	sudo sh 00-buildImage.sh
	
The order to call the commands is the one presented by the numbers.
The script 00 only has to be run in the case of changing the script.py or requirements

The normal flow of the app is to call:
01-runContainer.sh to run
02-stopAndRemoveContainer.sh to delete the image.
