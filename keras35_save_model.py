#0.넘파이 라이브러리 불러오기
import numpy as np

#1.데이터 불러오기 및 데이터 구성 =>

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
#LSTM은 레이어층 불러올 때 사용

model=Sequential()
model.add(LSTM(200,input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

#이렇게 데이터없이 모델구성만 해도 가능함

#3.모델 저장
model.save("./model/save_keras35.h5") 
#.은 