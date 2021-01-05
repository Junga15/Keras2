#제대로 전처리(validation_data): 원칙적으로는 validation data가 좋다.
#다시 제대로 해야함

import numpy as np 

from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) 

print("========================")
print(x[:5]) 
print(y[:10]) 
print(dataset.feature_names)

from sklearn.model_selection import train_test_split

#train:test:val=6:2:2로 분리해보자. 1.먼저 train과 test로 나누고 2.test내에서 다시 test와 val을 나눈다.
#1.data set분리 train:test=6:4
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=False,train_size=0.6,random_state=66) 
#1.data set분리 test:validation=5:5
x_test,x_val,y_test,y_val=train_test_split(x_test,y_test,shuffle=False,test_size=0.5,random_state=66)

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
#model.add(Dense(128,activation='relu',input_dim=13)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
model.add(Dense(128,activation='relu',input_shape=13,)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2,validation_data=(x_val,y_val)) #validation split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2
#배치 사이즈가 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=1) #loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) #y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
#print(y_predict)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)



