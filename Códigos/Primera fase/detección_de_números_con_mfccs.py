# -*- coding: utf-8 -*-
"""Detección de números con MFCCS

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wNBlaT76uG_f0SUFSxisl_K4ZiH_tFGZ
"""

!pip uninstall -y scikit-learn
!pip uninstall -y pandas
!pip uninstall -y pandas_ml

!pip install scikit-learn==0.20.0
!pip install pandas==0.24.0
!pip install pandas_ml

!pip install --upgrade wandb
!wandb login 3192017bec404c8c4f881ace5da7c4d4debf8b93

# Commented out IPython magic to ensure Python compatibility.
try:
#   %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import librosa.display as dsp
import random as rn
import pandas as pd
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from keras import Input
from keras.engine import Model
from keras.utils import to_categorical
from keras.layers import TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, \
    LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback

! git clone https://github.com/Jakobovski/free-spoken-digit-dataset

DATA_DIR = '/content/free-spoken-digit-dataset/recordings/'

def extract_features_mfccs(wav, sample_rate):
  mfccs = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=50)
  mfccs_processed = np.mean(mfccs.T, axis=0)

  return mfccs_processed

features_mfccs = []

for fname in os.listdir(DATA_DIR):
  if '.wav' not in fname:
    continue

  struct = fname.split('_')
  digit = struct[0]
  speaker = struct[1]

  audio, sample_rate = librosa.load(DATA_DIR + fname, sr=None, res_type='kaiser_fast')

  data_mfccs = extract_features_mfccs(audio, sample_rate)

  features_mfccs.append([data_mfccs, digit])

features_Panda_mfccs = pd.DataFrame(features_mfccs, columns=['feature_mfccs','class_label'])

features_Panda_mfccs.head()

X_mfccs = np.array(features_Panda_mfccs.feature_mfccs.tolist())
y = np.array(features_Panda_mfccs.class_label.tolist())

yy = to_categorical(y)

x_train_mfccs, x_test_mfccs, y_train_mfccs, y_test_mfccs = train_test_split(X_mfccs, yy, test_size=0.2, random_state = 127)
print("X_train_mfccs shape : ", x_train_mfccs.shape)
print("Y train shape_mfccs : ", y_train_mfccs.shape)

x_test_mfccs, x_validation_mfccs, y_test_mfccs, y_validation_mfccs = train_test_split(x_test_mfccs, y_test_mfccs, test_size=0.5, random_state = 127)
print("X_test_stft shape : ", x_test_mfccs.shape)
print("Y_test_stft shape : ", y_test_mfccs.shape)
print("X_validation_stft shape : ", x_validation_mfccs.shape)
print("Y_validation_stft shape : ", y_validation_mfccs.shape)

num_labels = yy.shape[1]

model_mfccs = tf.keras.Sequential([
                             tf.keras.layers.Input(shape=(50,)),
                             tf.keras.layers.Dense(256, activation='swish'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(256, activation='swish'),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(num_labels, activation='softmax')
])

model_mfccs.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model_mfccs.summary()

score = model_mfccs.evaluate(x_test_mfccs, y_test_mfccs, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 32

wandb.init(project="tfg")

model_mfccs.fit(x_train_mfccs, y_train_mfccs, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test_mfccs, y_test_mfccs), verbose=1, callbacks=[WandbCallback()])

score = model_mfccs.evaluate(x_train_mfccs, y_train_mfccs, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))

score = model_mfccs.evaluate(x_test_mfccs, y_test_mfccs, verbose=0)
print("Testing Accuracy: {0:.2%}".format(score[1]))

score = model_mfccs.evaluate(x_validation_mfccs, y_validation_mfccs, verbose=0)
print("Validation Accuracy: {0:.2%}".format(score[1]))

"""**Creación Matriz de Confusión y estadísticas**"""

prediction = model_mfccs.predict_classes(x_validation_mfccs)
value = []

for i in range(0,300):
  value.append(np.where(y_validation_mfccs[i] == 1)[0][0])

from sklearn.metrics import confusion_matrix
import seaborn as sns
from pandas_ml import ConfusionMatrix as confusion_pandas

labels=[0,1,2,3,4,5,6,7,8,9]
figsize=(10,10)

cm = confusion_matrix(value, prediction, labels=labels)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = '0'
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)
cm = pd.DataFrame(cm, index=labels, columns=labels)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=figsize)
sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Greens")

confusion_pandas = confusion_pandas(value, prediction)
confusion_pandas.print_stats()

print("TPR")
print(confusion_pandas.TPR)
print("PPV")
print(confusion_pandas.PPV)
print("F1")
print(confusion_pandas.F1_score)

"""**Transformación a Tensorflow Lite**"""

import time

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_path_sm)
tflite_model = converter.convert()

import pathlib

tflite_model_file = pathlib.Path('deteccionNumeroSinCND_ConSoundfile.tflite')
tflite_model_file.write_bytes(tflite_model)

try:
  from google.colab import files
  files.download(tflite_model_file)
except:
  pass

"""**PRUEBA DEL ACCURACY DE LOS MODELOS TFLITE**"""

import time

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

import pathlib

tflite_model_file = pathlib.Path('deteccionNumeroSinCND.tflite')
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_path='deteccionNumeroSinCND.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print('input_shape:', input_shape)

count_true = 0
count_false = 0

for i in range(0,400):
  input_data = np.array([x_test[i]], dtype=np.float32)

  np.expand_dims(input_data, -1)

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])

  value = np.where(y_test[i] == 1)

  if(np.argmax(output_data) == value):
    #print("Prediction of ", i, " : ", np.argmax(output_data), " REAL VALUE : ", value)
    count_true += 1
  else:
    #print("False")
    count_false += 1

print("ACIERTOS : ", count_true)
print("FALLOS : ", count_false)