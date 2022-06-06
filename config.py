"""
a config file to store all the necessary variables
"""
from sklearn.model_selection import train_test_split
import numpy as np

is_colorful = True
if is_colorful:
    no_channels = 3
else:
    no_channels = 1

img_config = {
    'img_height': 100, 
    'img_width':100, 
    'no_channels': no_channels,
    'no_img': 2000
    }

model_config = {
    'no_epochs':30,
    'test_size':0.20,
    'ffn_layer_1':10,
    'ffn_layer_2':10,
    'filter_CNN':4,
    'kernel_size':3,
    'batch_size':128
}

def split_data(images, labels):
    """
    method to split data into training and testing data set

    :param images: np.ndarray, a numpy array of all the images
    :param labels: list, a list of all the labels
    :return [X_train, X_test, y_train, y_test]: list, a list of randomly shuffled training 
            and testing data set
    """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=model_config['test_size'], random_state=42)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return X_train, X_test, y_train, y_test
