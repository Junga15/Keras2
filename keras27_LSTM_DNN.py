#keras23_LSTM3_scale을 DNN으로 코딩 =>시퀀셜 모델로 구성함
#결과치 비교
import numpy as np

'''
[로스값,mae],예측값
[1.9383718967437744, 0.9893341064453125]
[[7.5414257]]
=>LSTM보다 성능나쁨,지표가 잘안나옴
'''

#1.데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],54
           [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape:",x.shape) #(13,3)
print("y.shape:",y.shape) #(3,)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(10,activation='relu',input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) 

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x,y,epochs=100,batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([5,6,7]) #(3,) 
x_pred=x_pred.reshape(1,3) 

result=model.predict(x_pred)
print(result)