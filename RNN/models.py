import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Bidirectional, TimeDistributed, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization


from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


class Lstm:
    def __init__(self, lstm_units=64, dropout_rate=0.2, epoch=10, batch_size=32, learning_rate=0.001, input_shape=(39,44), num_classes=30, model_path='models\\lstm.h5'):
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.model_path = model_path
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
                            Bidirectional(LSTM(self.lstm_units, return_sequences=True), input_shape=self.input_shape),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            LSTM(self.lstm_units, return_sequences=True),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            TimeDistributed(Dense(64, activation='relu')),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            TimeDistributed(Dense(32, activation='relu')),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            LSTM(self.lstm_units),
                            BatchNormalization(),
                            Dropout(self.dropout_rate),
                            Dense(self.num_classes, activation='softmax')
                        ])
        
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    
    def train(self, train_Dataset, val_Dataset):
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
        callbacks_list = [checkpoint, early]
        self.history = self.model.fit(train_Dataset.batch(batch_size=self.batch_size), validation_data=val_Dataset.batch(batch_size=self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        
        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)
    

class Gru:
    def __init__(self, gru_units=64, dropout_rate=0.2, epoch=10, batch_size=32, learning_rate=0.001, input_shape=(39,44), num_classes=30, model_path='models\\gru.h5'):
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.model_path = model_path
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
                        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=self.input_shape),
                        BatchNormalization(),
                        MaxPooling1D(pool_size=2),
                        Dropout(self.dropout_rate),
                        Bidirectional(GRU(self.gru_units, return_sequences=True)),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        Bidirectional(GRU(self.gru_units, return_sequences=True)),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        GRU(self.gru_units, return_sequences=True),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        GRU(self.gru_units, return_sequences=True),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        TimeDistributed(Dense(256, activation='relu')),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        TimeDistributed(Dense(128, activation='relu')),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        TimeDistributed(Dense(64, activation='relu')),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        GlobalMaxPooling1D(),
                        Dense(512, activation='relu'),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        Dense(256, activation='relu'),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        Dense(128, activation='relu'),
                        BatchNormalization(),
                        Dropout(self.dropout_rate),
                        Dense(self.num_classes, activation='softmax')
                    ])
        
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    
    def train(self, train_Dataset, val_Dataset):
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
        callbacks_list = [checkpoint, early]
        self.history = self.model.fit(train_Dataset.batch(self.batch_size), validation_data=val_Dataset.batch(self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        
        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)
    

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(embed_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class Transformer:
    def __init__(self, num_heads=2, num_layers=1, dropout_rate=0.2, epoch=10, batch_size=32, learning_rate=0.001, input_shape=(39,44), num_classes=30, model_path='models\\transformer.h5'):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.model_path = model_path
        self.model = self.build_model()
    
    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(self.dropout_rate)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(embed_dim=128, num_heads=self.num_heads, dropout_rate=self.dropout_rate)(x)

        x = GlobalMaxPooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])

        return model
    
    def train(self, train_Dataset, val_Dataset):
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
        callbacks_list = [checkpoint, early]
        self.history = self.model.fit(train_Dataset.batch(self.batch_size), validation_data=val_Dataset.batch(self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        
        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)