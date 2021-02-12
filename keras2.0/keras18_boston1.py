#.keras18_boston1.py 실습
#keras09_mlp1 모델과 동일

'''
전처리 전 사이킷런에서 제공하는 보스턴 집값 로스 외 지표
loss: 36.2190055847168
mae: 4.313408851623535
RMSE: 6.018222786749827
R2: 0.5568457825690605
'''
#1.데이터 구성
import numpy as np 
#import pandas as pd #주석처리 제거가능
from sklearn.datasets import load_boston 
#사이킷런에서 제공하는 보스턴 집값 데이터셋

dataset=load_boston()
x=dataset.data
y=dataset.target
#x,y=dataset.data,dataset.target

print("========================")
#print('======================')
print(x.shape) #(506,13) input_shape 13, input_dim 13
print(y.shape) #(506,) 

print(x[:5]) 
print(y[:10]) 
print(np.max(x),np.min(x)) 
print(dataset.feature_names)
print(dataset.DESCR)

print(np.max(x),np.max(x))
x=(x-np.min(x))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

print(x_test.shape) 
print(y_test.shape)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=13)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
#model.add(Dense(128,activation='relu',input_shape=(13,)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=8,validation_split=0.2) #validation split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2
#배치 사이즈가 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=8) #loss,mae는 a,b로 바뀌어도 상관없음
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
