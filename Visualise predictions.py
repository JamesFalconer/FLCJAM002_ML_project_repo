from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, save_model
from PIL import Image
from PIL import ImageOps
import numpy as np
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

model = load_model(
    filepath+"Model_kfold_spec_big_data7",
    custom_objects=None,
    compile=True
)

input_test = np.load("C:/users/james/Downloads/MLProj/Data_test_small/x.npy")
target_test = np.load("C:/users/james/Downloads/MLProj/Data_test_small/y.npy",allow_pickle=True)

target_test = LabelEncoder().fit_transform(target_test[:,0])

print(len(input_test))

a = [0,10,65,17,36,63,33,51,24,53,19,5,40,37,52]


for i in a:
    sample_input = input_test[i]
    sample_input_array = np.array([sample_input])
    
    image = Image.fromarray(input_test[i].astype('uint8'),'L')
    display(image)
    prediction = model.predict(sample_input_array, verbose=0)
    if prediction[0][0] == 1:
        prediction = "Circle"
    elif prediction[0][1] == 1:
        prediction = "Square"
        
    else:
        prediction = "Triangle"

    print("Model prediction: " + str(prediction) )

