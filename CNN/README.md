# Convolutional Neural Networks

This is the Convolutional Neural Network project from Deep Learning. 
In this project we work on Cifar10 dataset, coming from https://www.kaggle.com/competitions/cifar-10/overview. 
Our goal is to explore CNN architectures, their strengths and weaknesses and overall performance on computer vision tasks. 

The directory is split into files containing important classes, functions and variables (mostly with `.py` extension), 
files for training the networks and performing experiments (`.ipynb`), 
files storing parameters of saved models (`.pt`) and 
`.csv` files for storing results for Kaggle submission. 

Here we provide detailed description of each respective file:
1. `cnn_models.py` - containing classes of our custom neural networks as well as class of pretrained alexNet architecture on ImageNet. 
2. `datasets.py` - containing PyTorch Dataset and Dataloader class for Cifar-10 data, classes used for data augmentation, transformers and instances of original and augmented data (ready for imports to other files).
3. `hyperparameter_search.py` - containing classes used for hyperparameter search, such as Net_wrapper, Grid Search, Random Search and Weighted Random Search.
4. `ensemble.py` - containing ensemble class. Compatible with neural network classes implementing proper `predict` method.
5. `weighted_random_search.py`- file implementing weighted random search algorithm (not complete).
6. `dataset_overview.ipynb` - containing overview of Cifar-10 data and augmented data.
7. `hyperparameter_tuning.ipynb` - containing code used for tuning hyperparameters of CNN custom model.
8. `ensemble.ipynb` - containing ensemble class along with experiments, results and plots of ensemble performance on original data.
9. `results.ipynb` - containing results of experiments on hyperparameter tuning.
10. `train_val_split.ipynb` - file containing method of splitting folder with Cifar-10 images into training and validation parts.

Folders: 
1. `ensemble_data` - containing predictions for ensemble models
2. `saved_models` - containing parameters of saved Convolutional Neural Network models.
3. `kaggle_submissions` - containing `.csv` files with predictions on testing data, prepared for Kaggle submission
4. `training` - containing files used for training the CNNs.
