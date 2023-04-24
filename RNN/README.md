# Useful materials

1. https://www.youtube.com/watch?v=4_SH2nfbQZ8
2. https://arxiv.org/pdf/1503.04069.pdf
3. https://www.youtube.com/watch?v=eCvz-kB4yko&t=956s
4. https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

# Useful notes:

1. Data preprocessing:
- Cut `.wav` files into shorter pieces, by removing silence before and after word. Add optional padding with zeros to fill up to certain time.
- Normalization of input.
- Data augmentation: pitch shifting, speed variation, reverberation. 

2. Feature extraction:
- MFCC.
- delta and delta-delta coefficients (MFCC and deltas both suggested in [2]).
- MEL spectrogram (without MFCC extraction).

3. Network:
- Bidirectional LSTM component (might be better than unidirectional LSTM to capture dependencies in both directions).
- Random search for hyperparams (suggested in [2]).

REPO STRUCTURE

RNN
|_'dataset.py', - file containing preprocessed datasets
|
|_ 'ensemble.ipynb', - file with code to create ensamble model and submit results
|
|_ 'extracted_features', - folder with data
|
|_ 'feature_extraction.ipynb', - file with feature extracions
|
|_ 'final_model.ipynb', - file that contains all aggregated data in aggregated form
|
|_ 'lstm_silence_model.ipynb', - training model to predict silence
|
|_ 'MFCC.ipynb', ????
|
|_ 'models', - folder with final models
|
|_ 'models.py', - file containing classes for mdoels
|
|_ 'preprocessing.py', - file with data preprocessing 
|
|_ 'results', - folder with aggregated results from training
|
|_ 'results.ipynb', - file that keeps classes and confusion matrixs for final concatenated models
|
|_ 'samples', - folder with samples
|
|_ 'submissions', - folder with saved submissions
|
|_ 'test', - test data
|
|_ 'train', - training data
|
|_ 'Training', - folder with ipynb notebooks that were used to train models
