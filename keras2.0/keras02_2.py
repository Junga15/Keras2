
import numpy as np #네이밍 룰
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras import models
#from tensorflow import keras
from tensorflow.keras.layers import Dense

#1.데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([2,4,6,8,10,12,14,16,18,20])
x_test=np.array([101,102,103,104,105,106,107,108,109,110])
y_test=np.array([111,112,113,114,115,116,117,118,119,111])

x_predict=np.array([101,102,103])

#2.모델구성
model=Sequential()
#model=models.Sequential()
#model=keras.models.Sequential()
model.add(Dense(200,input_dim=1,activation='linear'))
model.add(Dense(222))
model.add(Dense(204))
model.add(Dense(206))
model.add(Dense(208))
model.add(Dense(210))
model.add(Dense(212))
model.add(Dense(214))
model.add(Dense(216))
model.add(Dense(218))
model.add(Dense(220))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1,batch_size=1)
model.fit(x_train,y_train,epochs=1,batch_size=1)
model.fit(x_train,y_train,epochs=1,batch_size=1)
model.fit(x_train,y_train,epochs=1,batch_size=1)
model.fit(x_train,y_train,epochs=1,batch_size=1)
#배치 사이즈가 예를들어 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로
#배치사이즈 디폴트는 32(기본적으로 32번씩 던져주라), 데이터가 큰 데이터에서 배치사이즈 1로 잡아주는 것은 미친짓임


#4.평가,예측
loss = model.evaluate(x_test,y_test,batch_size=100)
print('loss:',loss)

result=model.predict([x_predict])
print("result:",result)

