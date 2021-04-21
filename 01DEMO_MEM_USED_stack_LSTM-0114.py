#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,GRU, Input, ConvLSTM2D, Bidirectional,BatchNormalization,Bidirectional
from tensorflow.keras import Input
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math
import json
from IPython.core.pylabtools import figsize
from scipy.fft import fft, ifft

figsize(15, 7) 
DATASET_NAME  = 'TJNEWM_HEAP_MEM_USED_85%MAX_2021_Jan&Feb.csv'

#dataset_train = pd.read_csv('./dataset/THGBNKLAK1TS301_Memory Used (percentage)_Oct_2020.csv') 
dataset_train = pd.read_csv('./dataset/'+DATASET_NAME)
#dataset_test = pd.read_csv('./dataset/ATKH_Oplus_TWGKHHPSK1MSB04_memory_usage_2020_11.csv')
dataset_train = dataset_train.reindex(index=dataset_train.index[::-1])
#dataset_test = dataset_test.reindex(index=dataset_test.index[::-1])
training_set = dataset_train.iloc[:40920,3:4].values
testing_set = dataset_train.iloc[40920:,3:4].values
#plt.plot(training_set)


# In[2]:


def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized# Memory growth must be set before GPUs have been initialized
            print(e)


# In[3]:

#print(testing_set.shape)
solve_cudnn_error()


# In[4]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler,RobustScaler


#MinMaxScaler


#sc_train = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc_train.fit_transform(training_set)

#sc_test = MinMaxScaler(feature_range = (0, 1))
#testing_set_scaled = sc_test.fit_transform(testing_set)


#RobustScaler

sc_train = RobustScaler()
training_set_scaled = sc_train.fit_transform(training_set)

sc_test = RobustScaler()
testing_set_scaled = sc_test.fit_transform(testing_set)


# In[9]:

SLIDE_WIN_SIZE = 60
def gen_dataset(data,x_window_size,y_window_size):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        X_train.append(data[i-x_window_size:i, :])
        y_train.append(data[i:i+y_window_size, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    
    return X_train,y_train
    
X_train,y_train = gen_dataset(training_set_scaled,SLIDE_WIN_SIZE,1)
X_test,y_test = gen_dataset(testing_set_scaled,SLIDE_WIN_SIZE,1)


# In[10]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[359]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

rp = ReduceLROnPlateau(
    monitor='loss', 
    factor=0.1, 
    patience=5, 
    verbose=1, 
    mode='auto', 
    min_delta=0.0001, 
    cooldown=0, 
    min_lr=0
)
filepath = "./model/NXP_model/NXP_best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
mode='max')


# In[360]:


#from keras import losses
model = Sequential()


model.add(LSTM(128,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=False,unroll=False))
model.add(Dense(y_train.shape[1]))
optz_fun = Adam(learning_rate=1e-4)
model.compile(loss='mae', optimizer=optz_fun)

model.summary()


# In[361]:


# In[362]:


model.layers[0].get_weights()[1]


# In[363]:


EPOCH = 50
BATCH = 128




# In[364]:


history = model.fit(X_train, y_train, epochs=EPOCH, batch_size=BATCH, validation_data=(X_test,y_test),callbacks=[rp,checkpoint],shuffle=False,verbose=1)


# In[357]:


plt.plot(history.history['loss'],label="Train loss")
plt.plot(history.history['val_loss'],label="Test loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(0,10)
plt.legend(loc='upper left')
plt.show()



# In[358]:


predicted_traffic = model.predict(X_test)



#predicted_traffic = sc_train.inverse_transform(predicted_traffic)  # to get the original scale
predicted_traffic = sc_test.inverse_transform(predicted_traffic) 
#predicted_traffic = predicted_traffic.astype('int64')

#plot_predicted_traffic = np.reshape(predicted_traffic,(-1,1))


# In[326]:


show_data_count = 0

plt.plot(testing_set[SLIDE_WIN_SIZE+1:],color = 'red', label = 'Real MEM Used')
plt.plot(predicted_traffic,color = 'blue', label = 'Predicted MEM Used')

#plt.axhline(y=60, xmin=0, xmax=predicted_traffic.shape[0],label = 'upper')
#plt.axhline(y=30, xmin=0, xmax=predicted_traffic.shape[0],label = 'low')
plt.xlabel('Sequence')
plt.ylabel('24H Detection Trend Values')
plt.title(DATASET_NAME)
plt.legend()
plt.show()


# In[327]:


from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
from  datetime import datetime

now_time = datetime.now()     #獲取當前時間
now_time = now_time.strftime('%Y%m%d_%H%M%S')   #列印需要的資訊,依次是年月日,時分秒,注意字母大小寫
print ("\nResult:")

print("-------------------------------")
#MAEScore_array = np.mean(abs(testing_set[61:]-predicted_traffic))
MAEScore = mean_absolute_error(testing_set[SLIDE_WIN_SIZE+1:],predicted_traffic)
RMSEScore = math.sqrt(mean_squared_error(testing_set[SLIDE_WIN_SIZE+1:],predicted_traffic))
#print(MAEScore_array)

print('Test Score: %.2f MAE' % (MAEScore))
print('Test Score: %.2f RMSE' % (RMSEScore))
print("-------------------------------")

"""
anomaly_index = dataset_test.loc[dataset_test['value'] >= 60, ['timestamp', 'value']].index
anomaly_value = dataset_test.loc[dataset_test['value'] >= 60, ['value']].values
anomaly_MAEScore = mean_absolute_error(predicted_traffic[anomaly_index],anomaly_value)
anomaly_RMSEScore = math.sqrt(mean_squared_error(predicted_traffic[anomaly_index],anomaly_value))


print('Anomaly_Test Score: %.2f MAE' % (anomaly_MAEScore))
print('Anomaly_Test Score: %.2f RMSE' % (anomaly_RMSEScore))

print("-------------------------------")
"""
"""
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='./model_plot/LSTM_AWS_model.png',show_shapes=True)
"""
#tf.saved_model.save(model, "Zabbix")
model.save("./model/NXP_model/Zabbix_model.h5")


# In[195]:


diff_list = []
for i in range(predicted_traffic.shape[0]):
    #plt.plot(predicted_traffic[i:i+1][0]-predicted_traffic[i-1][0])
    #print(predicted_traffic[i]-predicted_traffic[i-1])
    
    diff_list.append(abs(predicted_traffic[i]-predicted_traffic[i-1]))
plt.plot(diff_list)
plt.show()

# In[29]:


plt.plot(predicted_traffic)
plt.show()
