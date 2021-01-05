#다:1 mlp

#실습 train 과 test를 분리해서 소스를 완성하시오.
#=> train_test_split이용★

import numpy as np
'''
x=np.array([range(100)]) #print(x)은 [[0~99]]
x=np.array([range(301,401)]) #print(x)은 [[301~400]]
x=np.array([range(1,101)]) #print(x)은 [[1~100]]
print(x)
'''
x=np.array([range(100),range(301,401),range(1,101)])
y=np.array(range(711,811))

x=np.transpose(x)
print(x)
print(x.shape) #(100,3) 
print(y.shape) #711~810 => #(100,)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#train_test_split쓸 때 train_size(혹은 test_size)는 행을 분할하여 랜덤으로 train,test로 나눈다.
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)
'''
x_train=np.array([range(100),range(301,401)])
x_test=np.array(range(1,101))
y_train=np.array(range(711,761))
y_test=np.array(range(766,811))
=> 내가 잘못한 것, 범위를 쪼개면 안됨
'''

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #(100,)이므로 나가는 칼럼은 1개


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

