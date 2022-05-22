from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

#change filepath to where models are stored
filepath = "C:/users/Downloads/MLProj/Models/"
small_model1 = load_model(
    filepath+"Model_hold_small_data2",
    custom_objects=None,
    compile=True
)
#edit path to where test data is stored
input_test = np.load("C:/users/Downloads/MLProj/Data_test_small/x.npy")
target_test = np.load("C:/users/Downloads/MLProj/Data_test_small/y.npy",allow_pickle=True)

target_test = LabelEncoder().fit_transform(target_test[:,0])

model = small_model1
print("holdout Model_1 from small dataset performance")
print("Number of test examples: "+str(len(input_test)))
loss, acc = model.evaluate(input_test, target_test, verbose=0)
print('Model accuracy on unseen test data: {:5.2f}%'.format(100 * acc))
print('------------------------------------------------------------------------')
