from build_autoencoder import Autoencoder
from build_autoencoder import load_tensor
from visualize_wind_map import visualize_wind_map

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def autoencoder_train():
    x_train_data, x_train_norm = load_tensor('data/train/tensor/')
    y_train_data, y_train_norm = load_tensor('data/train/tensor/')

    #x_train_data = np.array(load_image('data/train/tensor_car/'))
    #y_train_data = np.array(load_image('data/train/tensor_car/'))

    x_train_data, x_val, y_train_data, y_val = train_test_split(x_train_data, y_train_data, test_size=0.2, random_state=42)

    ae = Autoencoder(len(x_train_data[0]), len(x_train_data[0][0]), len(x_train_data[0][0][0]))

    ae.train_model(x_train_data, y_train_data, x_val, y_val, epochs=5, batch_size=200, plot=False)
    ae.autoencoder_model.save_weights('weights_autoencoder/ae_weights.h5f', overwrite=True)

    """
    pred = ae.eval_model(x_val)

    plt.imshow(x_val[0])
    plt.show()
    plt.close()

    plt.imshow(pred[0])
    plt.show()
    plt.close()
    """

    #visualize_wind_map('train', x_val[0:1])
    #visualize_wind_map('train', pred[0:1])

def load_image(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        img = cv2.imread(path + name)
        img = cv2.resize(img, (100, 50), interpolation = cv2.INTER_AREA)
        img = img / 255.0
        tensor_list.append(img)
    return tensor_list
