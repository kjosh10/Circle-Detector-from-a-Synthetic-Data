from config import *
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Model_Generator():
    """
    this class contains method for generating various machine learning model

    Attributes:
    model_type: str, a string mentioning using either feed-forward or Convolution
                neural network
    """

    def __init__(self, model_type):
        self.model_type = model_type

    
    def feed_forward_network_generator(self):
        """
        method to generate a feed forward network

        :return model_ffn: Sequential, a sequential feed forward model
        """

        # Model with 2 hidden layers
        model = tf.keras.Sequential()
        model.add(layers.Dense(model_config['ffn_layer_1'], input_shape=(img_config['img_height']*
                img_config['img_width']*img_config['no_channels'],), activation="relu"))
        model.add(layers.Dense(model_config['ffn_layer_2'], activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model


    def cnn_generator(self):
        """
        method to generate a CNN network

        :return model_ffn: Sequential, a sequential CNN model
        """

        model_input = tf.keras.Input(shape=(img_config['img_height'], img_config['img_width'],
                            img_config['no_channels']))
        x = layers.Conv2D(model_config['filter_CNN'], 3, activation="relu")(model_input)
        x = layers.MaxPooling2D(model_config['kernel_size'])(x)
        x = layers.Conv2D(model_config['filter_CNN']*2, 3, activation="relu")(x)
        x = layers.MaxPooling2D(model_config['kernel_size'])(x)
        x = layers.Flatten()(x)
        model_output = layers.Dense(1, activation="sigmoid")(x)
        model_CNN = tf.keras.Model(model_input, model_output)
        model_CNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model_CNN


    def generate_model(self):
        """
        method to generate the model

        :return model: Sequential, a sequential model
        """

        if self.model_type == 'fnn':
            return self.feed_forward_network_generator()
        else:
            return self.cnn_generator()


    def plot_loss_and_acc(self, model):
        """
        method to plot loss and accuracy along epochs

        :param model: Sequential, a sequential model
        """

        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].plot(model.history.epoch, model.history.history['loss'], label="train")
        ax[0].plot(model.history.epoch, model.history.history['val_loss'], label="test")
        ax[0].set_ylabel('loss')
        ax[0].set_xlabel('epochs')
        ax[0].set_title('Loss V/S Epochs')
        ax[0].legend()
        ax[1].plot(model.history.epoch, model.history.history['accuracy'], label="train")
        ax[1].plot(model.history.epoch, model.history.history['val_accuracy'], label="test")
        ax[1].set_ylabel('accuracy')
        ax[1].set_xlabel('epochs')
        ax[1].set_title('Accuracy V/S Epochs')
        ax[1].legend()
        