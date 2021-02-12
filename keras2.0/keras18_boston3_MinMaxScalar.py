#.keras18_boston3_MinMaxScalar.py 실습
'''
x전체를 한번에 minmaxscalar이용하여 전처리한 후
#트레인테스트분리 전 민맥스스칼라전처리 진행함
loss: 20.032238006591797
mae: 2.844045400619507
RMSE: 4.475738838983593
R2: 0.7548974441004621
'''
#<※주의사항>
#민맥스스칼라 전처리할 때 프레딕트도 트랜스폼해서 
#전처리 하지 않으면 프레딕트값 오류뜸(이혜지님 4천대나옴)
#3차원 이상은 전처리한 후 reshape해야함, 리쉡하고 전처리하면 안먹힘

#1.데이터 구성
import numpy as np 
#import pandas as pd
from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) #(506,13) 
print(y.shape) # (506,)
x=(x-np.min(x))/(np.max(x)-np.min(x)) #711.0.0.0=>1.0.0.0
print(np.max(x[0])) #0.99999999999999

#1.2.데이터 전처리
from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
print(np.max(x),np.min(x)) #395: 해당칼럼의 최대가 395

#1.3.트레인,테스트 분리
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) #행을 자르는 것
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)

print(x_test.shape) 
print(y_test.shape)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=13)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
#model.add(Dense(128,activation='relu',input_shape=(13,))) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=8,validation_split=0.2) 
#validation split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2
#배치 사이즈가 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로
#배치사이즈 디폴트는 32(기본적으로 32번씩 던져주라), 
#데이터가 큰 데이터에서 배치사이즈 1로 잡아주는 것은 미친짓임

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=8) 
#loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) 
#y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
#print(y_predict)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)


