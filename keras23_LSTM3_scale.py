#실습.LSTM 코딩하기,나는 80을 원하오.=>LSTM 1을 적용한 실습

import numpy as np

#1.데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape:",x.shape) #(13,3)
print("y.shape:",y.shape) #(3,)

#1.2 데이터 재구성 =>reshape
x=x.reshape(13,3,1) 

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) 

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([50,60,70]) #(3,) =>1행
x_pred=x_pred.reshape(1,3,1) 

result=model.predict(x_pred)
print(result)

'''
0.05104106664657593
[[80.62689]]
#몇번 돌려도 값이 비슷해야 좋은 모델임,실습을 많이 해봐야 감이늠,실습을 많이 해봐야 하는 이유
'''