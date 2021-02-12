<<<<<<< HEAD:keras2.0/keras26_LSTM_hamsu.py
#keras23_LSTM3_scale을 함수형(#2.모델링 구성,keras13_hamsu1로 땡겨옴,x_shape처음은 09mlp(퍼셉트론)부터함)으로 코딩
#DNN으로 23번 파일보다 로스를 좋게 만들 것
#민맥스스칼라 넣을 수 있으면 넣기
#민맥스스칼라 전처리할 때 프레딕트도 트랜스폼해서 전처리 하지 않으면 프레딕트값 오류뜸(이혜지님 4천대나옴)
#3차원 이상은 전처리한 후 reshape해야함, 리쉡하고 전처리하면 안먹힘

'''
로스값, mae, x예측값
[5.579379081726074, 1.7736308574676514],[[8.565919]]
=>LSTM보다 성능나쁨,지표가 잘안나옴
'''

import numpy as np

#1.데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape:",x.shape) #(13,3)
print("y.shape:",y.shape) #(3,)
x_pred=np.array([50,60,70]) #(3,) =>1행


#2.모델구성 =>함수형 모델로 만들기(keras13_hamsu1에서 끌어옴)

#2.1 함수형 모델
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.layers import Dense, Input
#from keras.layers import Dense
input1=Input(shape=(3,)) #인풋쉐이프값이 1이 아닌 3이다! 
dense1=Dense(5,activation='relu')(input1) #위에 input1이 dense1에 아웃풋이므로 뒤에 input1
dense2=Dense(3)(dense1)
dense3=Dense(4)(dense2)
outputs=Dense(1)(dense3)   #아웃풋레이어는 1
model=Model(inputs=input1,outputs=outputs)
model.summary()

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
=======
#keras23_LSTM3_scale을 함수형(#2.모델링 구성,keras13_hamsu1로 땡겨옴,x_shape처음은 09mlp(퍼셉트론)부터함)으로 코딩
#DNN으로 23번 파일보다 로스를 좋게 만들 것
#민맥스스칼라 넣을 수 있으면 넣기
#민맥스스칼라 전처리할 때 프레딕트도 트랜스폼해서 전처리 하지 않으면 프레딕트값 오류뜸(이혜지님 4천대나옴)
#3차원 이상은 전처리한 후 reshape해야함, 리쉡하고 전처리하면 안먹힘

'''
로스값, mae, x예측값
[5.579379081726074, 1.7736308574676514],[[8.565919]]
=>LSTM보다 성능나쁨,지표가 잘안나옴
'''

import numpy as np

#1.데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape:",x.shape) #(13,3)
print("y.shape:",y.shape) #(3,)
x_pred=np.array([50,60,70]) #(3,) =>1행


#2.모델구성 =>함수형 모델로 만들기(keras13_hamsu1에서 끌어옴)

#2.1 함수형 모델
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.layers import Dense, Input
#from keras.layers import Dense
input1=Input(shape=(3,)) #인풋쉐이프값이 1이 아닌 3이다! 
dense1=Dense(5,activation='relu')(input1) #위에 input1이 dense1에 아웃풋이므로 뒤에 input1
dense2=Dense(3)(dense1)
dense3=Dense(4)(dense2)
outputs=Dense(1)(dense3)   #아웃풋레이어는 1
model=Model(inputs=input1,outputs=outputs)
model.summary()

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
>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras26_LSTM_hamsu.py
