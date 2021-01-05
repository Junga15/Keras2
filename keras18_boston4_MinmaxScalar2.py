#제대로 전처리(validation_split): 기준을 x_train만 잡고 전처리 진행 
#x_train은 fit,transform, x_val,x_test,x_predict는 transform만 진행


import numpy as np 

from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) #(506,13) input_shape 13
print(y.shape) #(506,)  output input_dim 1
#mlp keras09_mlp1과 동일

print("========================")
print(x[:5]) #인덱스가 0~4, 교육용데이터라 전처리가 되어있는 상태임
print(y[:10]) #[24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9]

#print(np.max(x),np.min(x)) #최대값과 최소값 구하는법
                           #최대값 711, 최소값0
print(dataset.feature_names) #컬럼명, s안쓰면 에러뜸
#print(dataset.DESCR)

#데이터 전처리(MinMax) =>당연히 전처리 해야하는 것, 통상적으로 전처리하면 성능은 향상됨
#x=x/711.#파이썬은 형변환이 자연스러움, 711뒤에 점이 붙는 이유는 x를 실수형으로 표현하겠다는 것.
#x가 0부터 시작한 것을 알았기 때문에 위의 식, 만약 몰랐다면
#x=(x-최소)/(최대-최소)
# x=(x-np.min(x))/(np.max(x)-np.min(x)) #711.0.0.0=>1.0.0.0
# print(np.max(x[0])) #0.99999999999999
#print(np.max(x),np.min(x))


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
# scaler=MinMaxScaler()
# scaler.fit(x)
# x=scaler.transform(x)
# print(np.max(x),np.min(x)) #395: 해당칼럼의 최대가 395 #711.0.0.0=>1.0.0.0

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=13)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
#model.add(Dense(128,activation='relu',input_shape=13)) #데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
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

'''
loss: 17.45690155029297
mae: 2.76603364944458
RMSE: 4.178145526626401
R2: 0.7864077500859118 #값이 심하게 달라지면 파라미터 튜닝을 잘못한 것.
'''

