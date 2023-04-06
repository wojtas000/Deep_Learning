# Useful materials

1. https://www.youtube.com/watch?v=4_SH2nfbQZ8
2. https://arxiv.org/pdf/1503.04069.pdf
3. https://www.youtube.com/watch?v=eCvz-kB4yko&t=956s
4. https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

# Useful notes:

1. Data preprocessing:
- Cut `.wav` files into shorter pieces, by removing silence before and after word). Add optional padding with zeros to fill up to certain time.
- Normalization of input
- Data augmentation: pitch shifting, speed variation, reverberation. 

2. Feature extraction:
- MFCC
- delta and delta-delta coefficients (MFCC and deltas both suggested in [2])
- MEL spectrogram (without MFCC extraction)

3. Network:
- Bidirectional LSTM component (might be better than unidirectional LSTM to capture dependencies in both directions)
- Random search for hyperparams (suggested in [2])