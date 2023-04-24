# RNN directory structure

## Folders:

1. `extracted_features` - folder for preprocessed data - after VAD, padding, resampling and feature extraction
2. `models` - folder with final models. The model which performed best on Kaggle is `best_gru_label_vs_unknown.h5`. We further build an ensemble with this model as base classifier and achieve even better score. The pre-trained models for ensemble are in `models\\ensemble` directory.
3. `results` - folder with aggregated results from training
4. `samples` - folder with samples from training data (original and processed)
5. `submissions` - folder with saved submissions
6. `Training` - folder with ipynb notebooks that were used to train models (grid search + random search)


## `.py` files

1. `dataset.py` - file containing classes for pyTorch and Tensorflow datasets, along with instances of this datasets to import
2. `models.py` - file containing classes for models 
3. `preprocessing.py` - file containing funcions and classes devoted to preprocessing of data


## `.ipynb` files

1. `ensemble.ipynb` - file with code to create ensamble model and submit results
2. `feature_extraction.ipynb` - file used for extracting features and saving them to `extracted_features` folder
3. `final_model.ipynb` - file that contains all aggregated data in aggregated form
4. `lstm_silence_model.ipynb` - training model to predict silence
5. `MFCC.ipynb` - file giving general overwiev of the Speech Commands Dataset along with example preprocessing and feature extraction
6. `results.ipynb` - file that keeps classes and confusion matrices for final models

