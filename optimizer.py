import numpy as np
from keras.optimizers import Adam

def opt_classifier(param):
    return Adam(lr=param["lr_classifier"], beta_1=param["b1_classifier"], beta_2=param["b2_classifier"])

def opt_discriminator(param):
    return Adam(lr=param["lr_discriminator"], beta_1=param["b1_discriminator"], beta_2=param["b2_discriminator"])

def opt_combined(param):
    return Adam(lr=param["lr_combined"], beta_1=param["b1_combined"], beta_2=param["b2_combined"])

