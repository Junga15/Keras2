#<keras29_LSTM_ensemble2>
#1.데이터 구성
import numpy as np

from numpy import array 

x1=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

y=array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

print("x1.shape:",x1.shape) #(13,3)
print("x2.shape:",x2.shape) #(13,3)
print("y.shape:",y.shape) #(13,)

x1=x1.reshape(x1.shape[0],x1.shape[1],1) 
#x2=x2.reshape(x2.shape[0],x2.shape[1],1)
print(x1.shape) #(13, 3, 1)
#print(x2.shape) #(13, 3, 1)

#2. 모델 구성

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import LSTM,Dense,Input 

#모델1_LSTM모델으로 구성
input1=Input(shape=(3,1)) 
lstm1=LSTM(10,activation='relu')(input1) 
dense1=Dense(10, activation='relu')(lstm1)

#모델2_Dense모델로 구성
input2=Input(shape=(3,))  #(13,3)
dense2=Dense(10,activation='relu')(input2)
dense2=Dense(10,activation='relu')(dense2)
dense2=Dense(1, activation='relu')(dense2)

#모델병합concatenate
from tensorflow.keras.layers import concatenate
merge1=concatenate([dense1,dense2])

#중간층 모델구성
middle1=Dense(15)(merge1)
middle1=Dense(15)(middle1)

#모델 분기1
output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(1)(output1) 

#모델 선언
model=Model(inputs=[input1,input2], outputs=output1)

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit([x1,x2],y,epochs=10,verbose=1) 

#4.평가, 예측

result=model.evaluate([x1,x2],y)  
print(result)

y_pred=model.predict([x1_predict,x2_predict])
print(y_pred)