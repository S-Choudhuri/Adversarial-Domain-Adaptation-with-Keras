import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.applications.resnet50 import ResNet50

def build_embedding(inp):
    #conv1 = Conv2D(32, 3, activation='relu', name = 'e_conv1')(inp)
    #pool1 = MaxPool2D(3, name = 'e_pool1')(conv1)
    #conv2 = Conv2D(64, 3, activation='relu', name = 'e_conv3')(pool1)
    #conv2 = Conv2D(64, 3, activation='relu', name = 'e_conv2')(conv2)
    #pool2 = MaxPool2D(3, name = 'e_pool4')(conv2)
    #conv2 = Conv2D(64, 3, activation='relu', name = 'e_conv4')(pool2)
    #pool2 = MaxPool2D(3, name = 'e_pool2')(conv2)

    feat = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=([224, 224, 3]))
    pool2 = feat(inp)
    #pool2 = Flatten()(feat)

    return pool2

def build_classifier(param, embedding):
    flat = Flatten(name='c_flatten')(embedding)
    dense1 = Dense(400, activation='relu', name='c_dense1')(flat)
    dense1 = Dense(100, activation='relu', name='c_dense11')(dense1)
    dense2 = Dense(param["source_label"].shape[1], activation='softmax', name='c_dense2')(dense1)
    return dense2

def build_discriminator(embedding):
    flat = Flatten(name='d_flatten')(embedding)
    dense1 = Dense(400, activation='relu', name='d_dense1')(flat)
    dense1 = Dense(100, activation='relu', name='d_dense11')(dense1)
    dense2 = Dense(1, activation='sigmoid', name='d_dense2')(dense1)
    return dense2

def build_combined_classifier(inp, classifier):
    comb_model = Model(inputs=inp, outputs=[classifier])
    return comb_model

def build_combined_discriminator(inp, discriminator):
    comb_model = Model(inputs=inp, outputs=[discriminator])
    return comb_model

def build_combined_model(inp, comb):
    comb_model = Model(inputs=inp, outputs=comb)
    return comb_model
    