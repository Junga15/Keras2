#다:1 mlp 함수형
#keras10_mlp2.py를 함수형으로 바꾸시오. 

import numpy as np

x=np.array([range(100),range(301,401),range(1,101)])
y=np.array(range(711,811))

x=np.transpose(x)
print(x)
print(x.shape) #(100,3) 
print(y.shape) #711~810 => #(100,)

#2.모델구성

#2.1 시퀀셜 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #(100,)이므로 나가는 칼럼은 1개

#2.2 함수형 모델
from tensorflow.keras.models import Sequential,Model 
from tensorflow.keras.layers import Dense, Input

input1=Input(Shape=(1,))
dense1=Dense(5,activation='relu')(input1)
dense2=Dense(3)(dense1)
dense3=Dense(4)(dense2)
outputs=Dense(1)(dense3)
model=Model(inputs=input1,outputs=outputs)
models.summary()


#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
print(y_predict)

loss: 0.01810389757156372
mae: 0.1155942901968956

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

