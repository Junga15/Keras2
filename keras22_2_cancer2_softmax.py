#keras22_cancer1.py를 다중분류로 코딩하세요. 

import numpy as np
from sklearn.datasets import load_breast_cancer

datasets=load_breast_cancer()
x=datasets.data #실제 업무에서 데이터셋을 분석해야 모델링이 가능함
y=datasets.target

#1.데이터 전처리 y값에 대한 원핫인코딩,train test split,minmax하기

from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
y=to_categorical(y)
#y_train=to_categorical(y_train)
#y_test=to_categorical(y_test)
print(y) 
print(x.shape) #(569, 30)
print(y.shape) #(569, 2)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) 

print(x_test.shape) #(456, 30)
print(y_test.shape) #(456,2)

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
print(np.max(x),np.min(x)) #1.0000000000000002 0.0

#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(30,))) 
model.add(Dense(100,activation='relu'))                                   
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))                                                                            
model.add(Dense(2,activation='softmax')) 

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc','mae']) 
model.fit(x_train,y_train,epochs=10,validation_split=0.2)                              

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss:',loss) #[0.7790930271148682, 0.8771929740905762, 0.13554473221302032]

'''
y_predict=model.predict(x_test) 
print(y_predict)


print(x[-5:-1])
y_pred=model.predict(x[-5:-1])

#print(y_test)
print(y_pred)
print(y[-5:-1])

'''