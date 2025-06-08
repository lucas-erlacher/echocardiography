# setup
conda was used in this project to manage packages. the details of the conda environment used in this project are listed in /environment.yml


# folders
/cluster:       collection of SLURM scripts required to run training, fine-tuning, etc.
/heart_echo:    this code has been taken from https://gitlab.inf.ethz.ch/OU-VOGT/heart_echo/-/tree/master?ref_type=heads. bugs were fixed and additions were made (e.g. custom data augmentation steps).
/logs:          log directory of SLURM jobs 
/models:        directory containing all model code
/runs:          tensorboard logger logs information to folders in this directory
/scripts:       collection of python helper-scripts for finding optimal batch size, plotting, etc. 


# running code
1. configure model type and model hyperparameters in config.py
2. depending what code should be executed (training, fine-tuning, etc.) the respective SLURM script in /cluster needs to be executed. this will execute the respective python script on multiple cluster nodes in parallel each using a different seed.
3. logs of the SLURM job (e.g. console printouts or errors) will be written to /logs
4. logs of the tensorboard logger (see logger.py for contents of created tensorboards) will be written to /runs


# running evaluations
the primary way to evaluate a trained model is by configuring the evaluation settings in config.py followed by launching a training run using train.sh (the evaluation will automatically run after the training process completes).
if an evaluation should be performed on a pre-trained model without any further training one can configure the loading of this pretrained model in config.py and launch train.sh (which will then skip training an immediately proceed to evaluation).