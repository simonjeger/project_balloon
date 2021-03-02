import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Reshape, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D

class Autoencoder():
    def __init__(self, size_x, size_z, size_c):
        self.img_shape = (size_x, size_z, size_c)

        optimizer = Adam(lr=0.001)
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()

        self.half_model = self.build_half_model()
        self.half_model.compile(loss='mse', optimizer=optimizer)

    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        self.n_bottleneck = 128

        # encoder
        self.ec_c = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        self.ec_p = MaxPooling2D((2, 2), padding='same')(self.ec_c)
        self.ec_f = Flatten()(self.ec_p)
        self.ec_bn = Dense(self.n_bottleneck)(self.ec_f)

        # decoder
        self.dc_d = Dense(self.ec_f.shape[1])(self.ec_bn)
        self.dc_r = Reshape(self.ec_p.shape[1::])(self.dc_d)
        self.dc_c = Conv2D(self.ec_c.shape[-1], (3, 3), activation='relu', padding='same')(self.dc_r)
        self.dc_u = UpSampling2D((2, 2))(self.dc_c)
        output_layer = Conv2D(self.img_shape[-1], (3, 3), activation='sigmoid', padding='same')(self.dc_u)

        return Model(input_layer, output_layer)

    def build_half_model(self):
        input_layer = Input(shape=self.img_shape)

        # only encode this time, last layer is compressed version
        self.ec_c = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        self.ec_p = MaxPooling2D((2, 2), padding='same')(self.ec_c)
        self.ec_f = Flatten()(self.ec_p)
        output_layer = Dense(self.n_bottleneck)(self.ec_f)

        return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20, plot=False):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping])
        if plot:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
            plt.close()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

    def compress(self, x_test_data, x_test_norm, train_or_test):
        preds = self.half_model.predict(x_test_data)
        for n in range(len(preds)):
            # torch = [mean, std, nodes_in_bottle_neck_layer]
            torch.save(np.concatenate((x_test_norm[n], preds[n]), axis=0), 'data/' + train_or_test + '/tensor_comp/wind_map' + str(n).zfill(5) + '.pt')

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    norm_list = []
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        mean = np.mean(tensor)
        std = np.std(tensor)
        tensor = (tensor - mean)/std #normalize
        norm_list.append([mean, std])
        tensor_list.append(tensor)
    return [np.array(tensor_list), np.array(norm_list)]
