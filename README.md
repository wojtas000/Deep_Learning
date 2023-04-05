# Deep Learning

This is the official repository for projects from Deep Learning. 
The repository is split into directories, each containing files for other, separate project.

## Convolutional Neural Networks

`CNN` is folder for Convolutional Neural Network project from Deep Learning. The directory is split into files containing important classes, functions and variables (mostly with `.py` extension), files for training the networks and performing experiments (`.ipynb`), files storing parameters of saved models (`.pt`) and `.csv` files for storing results for Kaggle submission. 

Here we provide detailed description of each respective file:
1. `CNNpy.py` - containing classes of our custom neural networks as well as class of pretrained alexNet architecture on ImageNet. 
2. `datasets.py` - containing PyTorch Dataset and Dataloader class for Cifar-10 data, classes used for data augmentation, transformers and instances of original and augmented data (ready for imports to other files)
3. `hyperparameter_search.py` - containing classes used for hyperparameter search, such as Net_wrapper, Grid Search, Random Search and Weighted Random Search.
4. `weighted_random_search.py`- file implementing weighted random search algorithm (not complete)
5. `dataset_overview.ipynb` - containing overview of Cifar-10 data and augmented data
6. `hyperparameter_tuning.ipynb` - containing code used for tuning hyperparameters of CNN custom model
7. `ensamble.ipynb` - containing ensemble class along with experiments, results and plots of ensemble performance on original data
8. `ensamble_aug.ipynb` - containing ensemble class along with experiments, results and plots of ensemble performance on augmented data
9. `results.ipynb` - containing results of experiments on hyperparameter tuning.
10. `train model {1, 2, 3}.ipynb` - files used for simoultaneous training of CNN models
11. `trainAlexNet.ipynb`, `CNN3_full_train.ipynb` - files used for training AlexNet / CNN_3_class models
11. `train_val_split.ipynb` - file containing method of splitting folder with Cifar-10 images into training and validation parts.

Folders: 
1. `ensamble_data` - containing predictions for ensemble models
2. `saved_models` - containing parameters of saved Convolutional Neural Network models.
3. `submissions` - containing `.csv` files with predictions on testing data, prepared for Kaggle submission


## Recurrent Neural Networks

`RNN` is folder for Convolutional Neural Network project from Deep Learning. 

# Repository contributors:
- Mikołaj Zalewski
- Jan Wojtas
