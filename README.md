This repository contains the codebase for analyzing early training dynamics in Fully Connected Networks (FCNs). This includes multiple Jupyter notebooks and scripts to reproduce specific results.

Note: Ensure that various util scripts are in the same folder as the notebooks.

# Installation

Before running any of the notebooks, please ensure that all the necessary libraries are installed. To do this, you can run the following command:

pip install -r requirements.txt

If you are using Google colab, most of the libraries come pre-installed. 

Also, you may wanna edit the path in the notebook for the working directory.


# Notebooks Overview

1. `early_training_phase_diagram.ipynb`: This notebook reproduces the early training results for FCNs trained on CIFAR10 using MSE loss.

2. `early_training_phase_diagram_zeros.ipynb`: This notebook reproduces the early training results for FCNs where the last layer is initialized to zero.

The above notebooks require that the following util scripts are in the same directory:
* `model_utils.py`
* `train_mse_utils.py` / `'train_xent_utils.py`
* `data_utils.py`

Runtime: Each notebook takes about 45 mins to run on a NVIDIA V100 with 16 GB RAM.	

# Customization

Results for other architectures, loss functions, and datasets can be reproduced by making minor modifications to the existing notebooks. Here are some instructions for common customizations:

* To change the loss function to cross-entropy: Replace the value of loss `mse` with `xent` in the `Load Libraries` section of the notebook. 

* To change the network architecture: Modify the model definition in the `Model Definition` section. Use the models defined in model_utils.py. For Myrtle CNNs, the implementation works for myrtle5, myrtle7 and myrtle10. The current immplementation of Myrtle CNNs is for CIFAR-10. For MNIST and Fashion-MNIST, the last avg_pool on `line 338` has to be removed to incorporate for the smaller image size. 

* To use a different dataset: Modify the dataset value in the `Hyperparameters` section.

# Additional Results

Intermediate saturation results can be obtained by tweaking the code to measure the sharpness at step tau that satisfies `c tau = K`, where `K` is a constant described in the main text. 