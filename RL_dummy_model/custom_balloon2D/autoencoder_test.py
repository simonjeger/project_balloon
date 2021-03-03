from build_autoencoder import Autoencoder
from build_autoencoder import load_tensor
from build_autoencoder import window

import numpy as np

def autoencoder_test(tensor, center):
    window_size = 10

    x_train_window, mean, std = window(tensor, center)
    x_train_window = np.array([x_train_window])

    ae = Autoencoder(len(x_train_window[0]), len(x_train_window[0][0]), len(x_train_window[0][0][0]))
    ae.autoencoder_model.load_weights('weights_autoencoder/ae_weights.h5f')
    return ae.compress(x_train_window, mean, std)
