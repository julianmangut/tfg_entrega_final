#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import librosa
import numpy as np

from time import time


# In[2]:


interpreter = tf.lite.Interpreter(model_path="CNN2D_10Classes_1000.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print('input_shape:', input_shape)


# In[3]:


tiempo_inicial = time() 

audio, sample_rate = librosa.load('Jose_follow.wav', sr=None, res_type='kaiser_fast')
print("AUDIO : ",audio)
print("SAMPLE_RATE : ",sample_rate)
print("AUDIO SHAPE : ", audio.shape)


# In[4]:


pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_processed = pad2d(mfccs,40)


# In[5]:


print(mfccs_processed.shape)

data = mfccs_processed.reshape(1,40,40,1)

print(data.shape)


# In[6]:


input_data = np.array(data, dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print('CALCULADO : ', np.argmax(output_data))

tiempo_final = time() 
tiempo_ejecucion = tiempo_final - tiempo_inicial

print('El tiempo de ejecucion fue : ',tiempo_ejecucion)


# In[ ]:




