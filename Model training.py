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

# Run for specialised data
batch_size = 64
img_width, img_height, img_num_channels = 54, 54, 1
loss_function = sparse_categorical_crossentropy
no_classes = 3
no_epochs = 20
optimizer = Adam()
verbosity = 1
num_folds = 10

#load training data
input_train = np.load("C:/users/james/Downloads/MLProj/Data_spec_aug/x.npy")
target_train = np.load("C:/users/james/Downloads/MLProj/Data_spec_aug/y.npy",allow_pickle=True)

input_train, input_test, target_train, target_test = train_test_split(input_train, target_train, test_size=0.2)

target_train = LabelEncoder().fit_transform(target_train[:,0])
target_test = LabelEncoder().fit_transform(target_test[:,0])

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

training_metrics = []
# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
    best_score = 0
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(no_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=loss_function,
                    optimizer=optimizer,
                    metrics=['accuracy'])


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                  batch_size=batch_size,
                  epochs=no_epochs,
                  verbose=verbosity)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])  
    
    print("Saving model")
    save_model(model,"C:/users/james/Downloads/MLProj/Models/Model_kfold_spec_big_data"+str(fold_no),save_format='h5')
    print("Model saved")
    metric = history.history['accuracy'][-1]
    training_metrics.append(metric*100)
    
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')