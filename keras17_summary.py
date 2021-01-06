import numpy as np
import tensorflow as tf

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(6,input_dim=1,activation='linear'))
model.add(Dense(4,activation='linear'))
model.add(Dense(5))
model.add(Dense(2))

model.summary()

#실습2 + 과제
#ensemble 1,2,3,4,에 대해 서머리를 계산하고 
#이해한 것을 과제로 제출할 것
#layer를 만들때 'name'이란놈에 대해 확인하고 설명할 것
#name을 반드시 써야할 때가 있다. 그때를 말해라.