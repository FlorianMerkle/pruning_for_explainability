# pruning_for_explainability
This is the accompanying repository of the master's thesis with the title "How much pruning is too much? The effects of neural network pruning on explainability". The contents of this repository allow to recreate the results of our experiments.


## Setup
The repository includes a `setup.sh`. It creates the necessary folders, downloads the Imagenette dataset,
and installs all packages from `requirements.txt` by creating a new conda environment.


### Training
The training phase is started with `vgg_training.py`. The script takes two arguments when started in the command-line:
The path to the `training_config.json`, and the desired `run`. The `training_config.json` includes a single run or multiple
runs that each specify the training-parameters and can be selected.

After training is done, the trained model is saved in the folder `trained_imgnette_models`, and the training results in
`training_output.json`.

The `helpers.py` file includes multiple helper functions for the training run, data and model processing.


### Pruning
After a model has been trained, it can be run through an iterative pruning sequence using `vgg_pruning.py`. If one 
choses to prune with only a subset of the specified compression rates, these can be given as parameters to the script.
Otherwise, the compression rates `[2, 4, 8, 16, 32, 64]` are applied. After every compression rate, the model is 
trained until convergence. Before the script proceeds with the next compression rate, the current model is saved in the 
`pruned_models` folder.

The required pruning methods are saved in `pruning_methods.py`. Currently, local magnitude unstructured and local
random unstructured are implemented for convolutional layers. The classes can be easily extended to prune other layers 
or with other methods as well. After the pruning has been completed, a csv with all relevant information is saved in the `pruning_results` folder. Currently, the folder includes the pruning results of
our pruning procedure.


### Experiments
For the experiments, Grad-CAM is required. The `gradcam` folder includes a PyTorch implementation of Grad-CAM and a
`utils.py` script with some helper functions. The juypter notebook `experiments_prep.ipynb` includes all relevant
steps to create the image-setup for the Mechanical Turk experiments. 

The `experiments/data` folder includes a .zip with a subset of the Imagenette images and labels that we chose for
our experiments. Before using the `experiments_prep.ipynb` notebook, these must be unzipped in order to be read by
the notebook.

To analyze the results, the `experiments_results.ipynb` jupyter notebook includes all steps to load the Mechanical Turk
results and transfrom them to insights. Our results are included in the `experiments` folder and are shown in the notebook.
