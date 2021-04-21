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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,GRU, Input, ConvLSTM2D, Bidirectional,BatchNormalization
from tensorflow.keras import Input
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math
import json
from IPython.core.pylabtools import figsize
figsize(15, 7) 

#dataset_train = pd.read_csv('./dataset/THGBNKLAK1TS301_Memory Used (percentage)_Oct_2020.csv')  # 讀取訓練集
dataset_train = pd.read_csv('./dataset/ATKH_Oplus_TWGKHHPSK1MSB04_memory_usage_2020_10.csv')
dataset_test = pd.read_csv('./dataset/ATKH_Oplus_TWGKHHPSK1MSB04_memory_usage_2020_11.csv')# 讀取訓練集
dataset_train = dataset_train.reindex(index=dataset_train.index[::-1])
dataset_test = dataset_test.reindex(index=dataset_test.index[::-1])
training_set = dataset_train.iloc[:,3:4].values
testing_set = dataset_test.iloc[:,3:4].values


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


def gen_dataset(data,x_window_size,y_window_size):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        X_train.append(data[i-x_window_size:i, :])
        y_train.append(data[i:i+y_window_size, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    
    return X_train,y_train
    
X_train,y_train = gen_dataset(training_set_scaled,10,1)
X_test,y_test = gen_dataset(testing_set_scaled,10,1)


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
filepath=r".\model\NXP_model\NXP_best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
mode='max')


# In[360]:


#from keras import losses
model = Sequential()


model.add(GRU(60, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True,unroll=False))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(GRU(60, return_sequences=True,unroll=False))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(GRU(60,return_sequences=False,unroll=False))

#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(LSTM(60, return_sequences=False,unroll=False))
model.add(Dense(y_train.shape[1]))
optz_fun = Adam(learning_rate=0.001)
model.compile(loss='mae', optimizer=optz_fun)
model.summary()


# In[361]:


units = 60
W = model.layers[0].get_weights()[0]
U = model.layers[0].get_weights()[1]
b = model.layers[0].get_weights()[2]

W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]


U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]

b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]

print("{},\n{},\n{},\n{}\n".format(b_i,b_f,b_c,b_o))


# In[362]:


model.layers[0].get_weights()[1]


# In[363]:


EPOCH = 50
BATCH = 32


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


# In[135]:


model = load_model("./Model/NXP_model/NXP_best_model.h5")


# In[358]:


predicted_traffic = model.predict(X_test)



#predicted_traffic = sc_train.inverse_transform(predicted_traffic)  # to get the original scale
predicted_traffic = sc_test.inverse_transform(predicted_traffic) 
predicted_traffic = predicted_traffic.astype('int64')

#plot_predicted_traffic = np.reshape(predicted_traffic,(-1,1))


# In[326]:


show_data_count = 0

plt.plot(testing_set[60:],color = 'red', label = 'Real CPU Used')
plt.plot(predicted_traffic,color = 'blue', label = 'Predicted CPU Used')

#plt.axhline(y=60, xmin=0, xmax=predicted_traffic.shape[0],label = 'upper')
#plt.axhline(y=30, xmin=0, xmax=predicted_traffic.shape[0],label = 'low')
plt.xlabel('Sequence')
plt.ylabel('24H Detection Trend Values')
plt.legend()


# In[327]:


from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
from  datetime import datetime

now_time = datetime.now()     #獲取當前時間
now_time = now_time.strftime('%Y%m%d_%H%M%S')   #列印需要的資訊,依次是年月日,時分秒,注意字母大小寫
print ("Result:")

print("-------------------------------")
#MAEScore_array = np.mean(abs(testing_set[61:]-predicted_traffic))
MAEScore = mean_absolute_error(testing_set[61:],predicted_traffic)
RMSEScore = math.sqrt(mean_squared_error(testing_set[61:],predicted_traffic))
#print(MAEScore_array)

print('Test Score: %.2f MAE' % (MAEScore))
print('Test Score: %.2f RMSE' % (RMSEScore))

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
tf.saved_model.save(model, "Zabbix")
#model.save("./model/AWS_model/AWS_model.h5")


# In[195]:


diff_list = []
for i in range(predicted_traffic.shape[0]):
    #plt.plot(predicted_traffic[i:i+1][0]-predicted_traffic[i-1][0])
    #print(predicted_traffic[i]-predicted_traffic[i-1])
    
    diff_list.append(abs(predicted_traffic[i]-predicted_traffic[i-1]))
plt.plot(diff_list)


# In[29]:


plt.plot(predicted_traffic)


# In[28]:


predicted_traffic.shape


# In[29]:


print(log_data.shape)
log_data = np.expand_dims(log_data, axis=0)
print(log_data.shape)


# In[30]:


log_data[0][:,3:4].shape


# In[653]:


log_data


# In[654]:


log_data[-1][0]


# In[651]:


DATETIME = dt.strptime(log_data[-1][0]+'_'+log_data[-1][1],"%Y/%m/%d_%H:%M:%S")


# In[648]:


log_data[-1][3]


# In[673]:


from datetime import datetime as dt
from datetime import timedelta
#datetime.strptime("2018-01-31", "%Y-%m-%d")

log_data = dataset_train.iloc[:60].values
mem_data = dataset_train.iloc[:60,3:4].values
for i in range(60,dataset_train.shape[0],1):
    log_data = np.append(log_data,dataset_train.iloc[i:i+1].values,axis=0)
    log_data = np.delete(log_data,1,axis=0)
    DATETIME = dt.strptime(log_data[-1][0]+'_'+log_data[-1][1],"%Y/%m/%d_%H:%M:%S")
    sample = log_data[:,3:4]
    pred_scaler = RobustScaler()
    sample_scaled = pred_scaler.fit_transform(sample)
    
    sample_scaled = np.expand_dims(sample_scaled,axis=0)
    

    
    
    pred_data = model.predict(sample_scaled)
    pred_data = pred_scaler.inverse_transform(pred_data)
    diff = log_data[-1][3]-pred_data
    if abs(diff)  >= 30 or pred_data >= 60 :
        anomaly_time = DATETIME+timedelta(minutes=3)
        print("\nPredict anomaly time : "+anomaly_time.strftime("%Y/%m/%d_%H:%M:%S"))
        print("Anomaly value : {}".format(log_data[-1][3]) )
        with open('./anomaly_log_file/anomaly_data5.txt', 'a') as f:
            f.write("[{}]{}".format(anomaly_time.strftime("%Y/%m/%d_%H:%M:%S"),log_data[-1][3]))
            f.write("\n")
    """
pred_data = model.predict(mem_data)
diff = mem_data[0][-1][0]-pred_data
if abs(diff) >= 30:
    print(diff)
DATETIME = dt.strptime(log_data[-1][0]+'_'+log_data[-1][1],"%Y/%m/%d_%H:%M:%S")
print(DATETIME)
#TIME = datetime.strptime(log_data[-1][1],"%H:%M:%S")
#log_data[-1][0]+'_'+log_data[-1][1]
DATETIME+timedelta(minutes=3)
"""


# In[ ]:


temp_date = ""
anomaly_date_list = []
for i in range(X_data.shape[0]):
    
    
    pred_data = sc.inverse_transform(X_data[i])
    max_value = np.max(pred_data)
    if max_value >= 250 and time[i,np.argmax(pred_data)] != temp_date:
        anomaly_date_list.append(time[i,np.argmax(pred_data)])
        max_index = np.argmax(pred_data)
        print("Time : {}\nAnomaly Value : {}\n".format(time[i,max_index],max_value))
        with open('./anomaly_log_file/anomaly_data4.txt', 'a') as f:
            f.write("[{}]{}".format(time[i,max_index],max_value))
            f.write("\n")
        print(temp_date)
    temp_date = time[i,np.argmax(pred_data)]


# In[558]:



for i in range(dataset_train.shape[0]-22000):
    offset = abs(dataset_train.iloc[i:i+1,3:4].values-dataset_train.iloc[i+1:i+2,3:4].values)
    if offset >= 30:
        print(offset)


# In[538]:


anomaly_index = dataset_test.loc[dataset_test['%used'] >= 60, ['time', '%used']].index
anomaly_value = dataset_test.loc[dataset_test['%used'] >= 60, ['%used']].values

for i in anomaly_index:
    print(i)
    anomaly_MAEScore = mean_absolute_error(predicted_traffic[i],dataset_test.iloc[i:i+1,4:5].values)
    anomaly_RMSEScore = math.sqrt(mean_squared_error(predicted_traffic[i],dataset_test.iloc[i:i+1,4:5].values))


    #print('Anomaly_Test Score: %.2f MAE' % (anomaly_MAEScore))
    print('Anomaly_Test Score: %.2f RMSE' % (anomaly_RMSEScore))
    
    


# In[158]:


for i in range(predicted_traffic.shape[0]):
    q_value = np.percentile(predicted_traffic[i:i+20], (25, 50, 99), interpolation='midpoint')
    if q_value[2] >=30:
        print(q_value[2])


# In[165]:


plt.boxplot(x=predicted_traffic,whis = 20,widths = 0.7,patch_artist = True,showmeans = True,boxprops = {'facecolor':'steelblue'}
            ,flierprops = {'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':4}
            ,meanprops = {'marker':'D','markerfacecolor':'black', 'markersize':4}
            ,medianprops = {'linestyle':'--','color':'orange'},labels = [''])



plt.show


# In[166]:


for i in range(predicted_traffic.shape[0]-20):
    #print(i,i+20)
    max_value = np.max(np.percentile(predicted_traffic[i:i+20], (25, 50, 99), interpolation='midpoint'))
    if max_value >= 30:
        print(max_value)


# In[ ]:




