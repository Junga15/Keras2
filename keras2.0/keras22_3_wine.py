#DNN계열
#LSTM 구현과 비교 keras33_LSTM5_wine.py


'''
<처음에 임포트 먼저 다해줘도 상관없음> #김수현 19_1참고
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error
'''

import numpy as np 
from sklearn.datasets import load_wine
datasets=load_wine()
x=datasets. data
y=datasets.target 
print(x)
print(y) #[0000..11111...2222..]
print(x.shape) #(178,13)
print(y.shape) #(178,)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
#model.add(Dense(128,input_dim=13,activation='relu'))
model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(64))

model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,validation_split=0.2,epochs=200,batch_size=8,verbose=1)

y_predict=model.predict(x_test) 
#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE',RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

