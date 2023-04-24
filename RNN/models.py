import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Bidirectional, TimeDistributed, BatchNormalization, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization
import tensorflow.keras.backend as K

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


file_dir = os.path.dirname(os.path.abspath(__file__))

class Lstm:
    """
    LSTM model class.
    """
    def __init__(self, lstm_units=64, dropout_rate=0.2, epoch=10, batch_size=32, learning_rate=0.001, input_shape=(39,44), num_classes=30, model_path=os.path.join(file_dir, 'models\\lstm.h5'), from_path=False):
        """
        Args:
            lstm_units (int): Number of units in the LSTM layer.
            dropout_rate (float): Dropout rate.
            epoch (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            input_shape (tuple): Input shape.
            num_classes (int): Number of classes.
            model_path (string): Path to save the model.
            from_path (string): Path to load the model from.
        """

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
        self.from_path=from_path
        
        if self.from_path:
            self.model = tf.keras.models.load_model(self.from_path)
        else:
            self.model = self.build_model()
    
    def build_model(self):
        """
        Build the model.
        Returns:
            model (keras.model): The model.
        """

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
    
    def train(self, train_Dataset, val_Dataset = None):
        """
        Train the model.
        Args:
            train_Dataset (tf.data.Dataset): Training dataset.
            val_Dataset (tf.data.Dataset): Validation dataset.
        """

        if val_Dataset is None:
            checkpoint = ModelCheckpoint(self.model_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(batch_size=self.batch_size), epochs=self.epoch, callbacks=callbacks_list)
        else:    
            checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(self.batch_size), validation_data=val_Dataset.batch(self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        """
        Predict the labels of the test dataset.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.array: Predicted labels.
        """

        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)
    

class Gru:
    """
    GRU model class.
    """
    def __init__(self, gru_units=64, dropout_rate=0.2, epoch=10, batch_size=32, learning_rate=0.001, input_shape=(39,44), num_classes=30, model_path=os.path.join(file_dir, 'models\\gru.h5'), from_path=False):
        """
        Args:
            gru_units (int): Number of units in the GRU layer.
            dropout_rate (float): Dropout rate.
            epoch (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            input_shape (tuple): Input shape.
            num_classes (int): Number of classes.
            model_path (string): Path to save the model.
            from_path (string): Path to load the model from.
        """
        
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
        self.from_path=from_path
        
        if self.from_path:
            self.model = tf.keras.models.load_model(self.from_path)
        else:
            self.model = self.build_model()
    
    def build_model(self):
        """
        Build the model.
        Returns:
            model (keras.model): The model.
        """

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
    
    def train(self, train_Dataset, val_Dataset = None):
        """
        Train the model.
        Args:
            train_Dataset (tf.data.Dataset): Training dataset.
            val_Dataset (tf.data.Dataset): Validation dataset.
        """

        if val_Dataset is None:
            checkpoint = ModelCheckpoint(self.model_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(batch_size=self.batch_size), epochs=self.epoch, callbacks=callbacks_list)
        else:    
            checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(self.batch_size), validation_data=val_Dataset.batch(self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        """
        Predict the labels of the test dataset.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.array: Predicted labels.
        """

        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)
    

class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block class.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate, **kwargs):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of heads in attention.
            dropout_rate (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = Sequential([
            Dense(self.embed_dim, activation='relu'),
            Dense(self.embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
    
    def get_config(self):
        """
        Get the config of the model.
        Returns:
            config (dict): The config of the model.
        """
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, inputs):
        """
        Call the model.
        Args:
            inputs (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Output tensor.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class Transformer:
    """
    Transformer class.
    """
    def __init__(self, num_heads=2, 
                 num_layers=1, 
                 dropout_rate=0.2, 
                 epoch=10, 
                 batch_size=32, 
                 learning_rate=0.001, 
                 input_shape=(39,44), 
                 num_classes=30, 
                 model_path=os.path.join(file_dir, 'models\\transformer.h5'),
                 from_path=False):
        """
        Args:
            num_heads (int): Number of heads in attention of transformer block.
            num_layers (int): Number of layers of transformer block.
            dropout_rate (float): Dropout rate.
            epoch (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            input_shape (tuple): Input shape.
            num_classes (int): Number of classes.
            model_path (str): Path to save the model.
            from_path (str): Path to load the model.
        """


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
        self.from_path=from_path
        
        if self.from_path:
            model = tf.keras.models.load_model(self.from_path, custom_objects=self.custom_objects())
            self.model = model
        
        else:
            self.model = self.build_model()
    
    def build_model(self):
        """
        Build the model.
        Returns:
            tf.keras.Model: The model.
        """
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
    
    def train(self, train_Dataset, val_Dataset = None):
        """
        Train the model.
        Args:
            train_Dataset (tf.data.Dataset): Training dataset.
            val_Dataset (tf.data.Dataset): Validation dataset.
        """
        if val_Dataset is None:
            checkpoint = ModelCheckpoint(self.model_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(batch_size=self.batch_size), epochs=self.epoch, callbacks=callbacks_list)
        else:    
            checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)
            callbacks_list = [checkpoint, early]
            self.history = self.model.fit(train_Dataset.batch(self.batch_size), validation_data=val_Dataset.batch(self.batch_size), epochs=self.epoch, callbacks=callbacks_list)

    def predict(self, test_Dataset):
        """
        Predict the model.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        return np.argmax(self.model.predict(test_Dataset.batch(10)), axis=1)
    
    def custom_objects(self):
        """
        Get the custom objects of the model.
        Returns:
            dict: Custom objects.
        """
        return {"TransformerBlock": TransformerBlock}
    
    
if __name__=='__main__':
    from dataset import label_detection_training, label_detection_validation
    model = Transformer(epoch=1, num_classes=11, model_path='models\\transformer.h5')
    model.train(label_detection_validation, label_detection_validation)
    #save model
    model2 = Transformer(from_path='models\\transformer.h5')
    model2.model.summary()


class Ensemble:
    """
    Ensemble of models.
    """
    def __init__(self, model_paths):
        """
        Args:
            model_paths (list): Paths to models.
        """
        self.models = []
        for model in model_paths:
            self.models.append(tf.keras.models.load_model(model))
    
    def predict_mean(self, test_Dataset):
        """
        Predict the model using mean method.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(test_Dataset))
        predictions = np.array(predictions)
        return np.argmax(np.mean(predictions, axis=0), axis=1)
 
    
    def predict_max(self, test_Dataset):
        """
        Predict the model using max method.
        Args:
            test_Dataset (tf.data.Dataset): Test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(test_Dataset))
        predictions = np.array(predictions)
        return np.argmax(np.max(predictions, axis=0), axis=1)
      