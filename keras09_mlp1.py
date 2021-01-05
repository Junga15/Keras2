#다:1 mlp

import numpy as np

x=np.array([[1,2,3,4,5,6,7,8,9,10],
           [11,12,13,14,15,16,17,18,19,20]])
y=np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #(10,): 스칼라(행속에 있는 변수)가 10개
               #행추가(x=[[1~10],[1~10]]): (2,10) 스칼라가 20개, 칼럼/열/특성이 10개

x=np.transpose(x)
print(x)
print(x.shape) #=>칼럼이 2개있는 (10,2)로 바꾸기 

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x,y,epochs=100,batch_size=1,validation_split=0.2) #validation split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2

 #4.평가,예측
loss,mae=model.evaluate(x,y) #loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x) #y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
#print(y_predict)

'''
loss: 0.01810389757156372
mae: 0.1155942901968956
'''
