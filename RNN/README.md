# Recurrent Neural Networks

This is the Recurrent Neural Network project from Deep Learning. 
In this project we work on Tensorflow Speech Recognition challenge, coming from https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge. 
Our goal is to explore RNN architectures, Transformers and select best performing models for speech detection.

# Directory structure

## Folders:

1. `report_and_presentation` - folder containing report and presentation of RNN project
2. `extracted_features` - folder for preprocessed data - after VAD, padding, resampling and feature extraction
3. `models` - folder with final models. The model which performed best on Kaggle is `best_gru_label_vs_unknown.h5`. We further build an ensemble with this model as base classifier and achieve even better score. The pre-trained models for ensemble are in `models\\ensemble` directory.
4. `results` - folder with aggregated results from training
5. `samples` - folder with samples from training data (original and processed)
6. `submissions` - folder with saved submissions
7. `Training` - folder with ipynb notebooks that were used to train models (grid search + random search)


## `.py` files:

1. `dataset.py` - file containing classes for pyTorch and Tensorflow datasets, along with instances of this datasets to import
2. `models.py` - file containing classes for models 
3. `preprocessing.py` - file containing funcions and classes devoted to preprocessing of data


## `.ipynb` files:

1. `example.ipynb` - example file with code to create best GRU model (ensamble model of best GRU) and submit results
2. `feature_extraction.ipynb` - file used for extracting features and saving them to `extracted_features` folder
3. `final_model.ipynb` - file that contains all aggregated data in aggregated form
4. `lstm_silence_model.ipynb` - training model to predict silence
5. `MFCC.ipynb` - file giving general overwiev of the Speech Commands Dataset along with example preprocessing and feature extraction
6. `results.ipynb` - file that keeps classes and confusion matrices for final models

# Reproducing results

In order to reproduce best result on testing data from Kaggle you should:

1. Download `.zip` file from the following link: https://drive.google.com/file/d/1DVdjHTMePQvo_fouEO17QIcfPPstDpmm/view?usp=sharing. It contains pickle file of list of detected silence and pickle file containing list of features for testing data (due to large size of these files we do not host them on Github Repository)
2. Unpickle both files, save list of features of test data as tensorflow.data.Dataset 
3. Load `model\\best_gru_label_vs_unknown.h5` model.
4. Predict labels for test data with model and preprare kaggle submission

The example of such submission can be found in `example.ipynb` file (for submitting result for GRU and ensemble)

