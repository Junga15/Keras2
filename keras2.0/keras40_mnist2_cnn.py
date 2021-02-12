#<인공지능계의 hello world라 불리는 mnist!!>

#실습! 완성하세요
#지표는 acc:  이상적 0.985이상

#응용
#y_test 10개와 y_pred 10개를 출력하시요. 

#y_test[:10]=(,,,,,,,,,)
#y_pred[:10]=

import numpy as np
import matplotlib.pyplot as plt #그래프,이미지를 보고 싶다는 것

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,) #다중분류->원핫인코딩,투카테고리컬
print(x_test.shape,y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0]) 
print(y_train[0]) #5
print("y_train[0]:",y_train[0])
print(x_train[0].shape) #(28, 28)

# #plt.imshow(x_train[0],'gray')
# plt.imshow(x_train[0])
# plt.show() #특성이 있을수록 밝은 색이 됨,흰색이 255 가장큼

#.민맥스칼라 전처리
x_train=x_train.reshape(60000,28,28,1).astype('float32')/255 #정수형을 실수형으로 바꾼다. 전처리함
#x_train=x_train.reshape(60000,28,28,1)/255. #이렇게 해도 상관없음
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255   #이렇게 하면 민맥스 안해도 됨
#np.min,np.max 확인
#x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

#원핫인코딩 하세요. 

from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder()
onehot.fit(y_train.reshape(-1,1))
y=onehot.transform(y_train.reshape(-1,1)).toarray()

#모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten 

model=Sequential()
model.add(Conv2D(filters=100,kernel_size=(2,2),padding='same',
                 strides=1,input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(9,(2,2))) 


#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,validation_split=0.2,epochs=100)

#4.평가,예측

loss,acc = model.evaluate(x_test, y_test, batch_size=1)
print('loss:', loss)
print('acc:', acc)

print(x_train[0]) 
print(y_train[0]) #5
print("y_train[0]:",y_train[0])

y_pred=model.predict(x_test)
print(y_pred)

print(y_test[:10])
print(y_pred[:10])