#1:다 mlp 함수형
#keras10_mlp6.py를 함수형으로 바꾸시오. 

#1:다 mlp
#실습:코드를 완성할 것
#mlp4처럼 predict값을 도출할 것

import numpy as np

x=np.array(range(100))
y=np.array([range(711,811),range(1,101),range(100)])
print(x.shape) #(3,100)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y) #(100,3)
print(x)
print(x.shape) #=>(100,3)

x_pred2=np.array([0,1,99])
print("x_pred2.shape:",x_pred2.shape) #(20,3)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=1)) #칼럼의 갯수 3
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #(100,3)이므로 나가는 칼럼은 3개

#2.2 함수형 모델
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

input1=Input(Shape=(1,))
dense1=Dense(5,activation='relu')(input1)
dense2=Dense(3)(dense1)
dense3=Dense(4)(dense2)
dense4=Dense(1)(dense3)
outputs=Dense(1)(dense4)
model=Model(inputs=input1,outputs=outputs)
model.summary()

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
#print(y_predict)

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

y_pred2=model.predict(x_pred2)
print(y_pred2)
'''
#[[ 6.9808752e+02  3.0145443e+01 -4.9315697e-01]
 [ 6.9928442e+02  3.0585827e+01  5.1326168e-01]
 [ 8.1657959e+02  7.3743965e+01  9.9144989e+01]]
 '''