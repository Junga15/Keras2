#.keras18_boston2_minmax.py 실습
#<minmaxscalar,민맥스스칼라> =>#1.2.데이터 전처리 참고
#scaling스케일링: 범위 조절하는 것
#데이터 전처리의 일종
#민맥스스칼라=노말라이제이션='정규화'(normalization,regulazation)=>normalization과 비슷
#스케일한 범위는 0이상 1이하 사이 =민맥스스케일링
#x=(x-np.min(x))/(np.max(x)-np.min(x)) #711.0.0.0=>1.0.0.0
#x=(x-최소)/(최대-최소)
#print(np.max(x[0])) #0.99999999999999
#0~711이면 나누기711하면 0~1까지
#맨앞값이 0이 아닌 경우, 맨앞값-최소값하면0이됨
#따라서 스케일한 범위는 0이상 1이하 사이 =민맥스스케일링

'''
#18_2 수동전처리 모델구성 =>약간 이상하고 잘못된 방식
트레인테스트분리전 x=(x-np.min(x))/(np.max(x)-np.min(x)) 넣어줌
(※19_2는 분리후 위식으로 전처리함)
loss: 53.93804931640625
mae: 5.579458236694336
RMSE: 7.344253227362471
R2: 0.3400460221561894
'''
#1.데이터의 구성
import numpy as np 
#import pandas as pd #주석처리 제거가능
from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target
print(x.shape) #(506,13) input_shape 13
print(y.shape) #(506,) 

#1.2.데이터 전처리(MinMaxScalar,민맥스스카라)
x=x/711.
#당연히 전처리 해야하는 것, 통상적으로 전처리하면 성능은 향상됨
#파이썬은 형변환이 자연스러움, 711뒤에 점이 붙는 이유는 x를 실수형으로 표현하겠다는 것.
#x가 0부터 시작한 것을 알았기 때문에 위의 식, 만약 몰랐다면
#x=(x-최소)/(최대-최소)
x=(x-np.min(x))/(np.max(x)-np.min(x))
#print(np.max(x),np.min(x))

#1.3.트레인과 테스트의 분리
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) 
#행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

print(x_test.shape) #(405, 13)
print(y_test.shape) #(405,)

'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
'''

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#위아래 순서 바뀌어도 됨

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=13)) 
#데이터가 2차원이기 때문에 인풋 딤,인풋 쉐이프 모두 가능
#model.add(Dense(128,input_shape=(13,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=8,validation_split=0.2) #validation split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2
#배치 사이즈가 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=8) 
#loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) 
#y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
#print(y_predict)

#RMSE와 R2값 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)


