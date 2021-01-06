#실습 validation_data를 만들것
#train_test_split를 사용할 것! => x_val,y_val을 train_test_split로 만들것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 

#1.데이터
x=np.array(range(1,101)) #1~100
y=np.array(range(1,101))


#x_train=x[:60] #값:1~60 => 순서 0번째부터 59번째까지 값: 1~60
#x_val=x[60:80] #값:61~80
#x_test=x[80:] #값:81~100
#list slicing 리스트 슬라이싱

#y_train=y[:60] #1~60 => from No.00 to 59 value: 1~60
#y_val=y[60:80] #61~80
#y_test=y[80:] #81~100

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                             train_size=0.8,shuffle=True)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
'''

#2.모델 구성
model=Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,validation_data=)

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss','mse')

y_predict=model.predict(x_test)
print(y_predict)


#<선생님 답안>
#shuffle=False
#loss=0.0028
#mae=0.052695

#shuffle=True
#loss=0.0008 =>훨씬 성능이 좋아짐
#mae:0.02119

#shuffle=False 
loss: 0.08563689887523651
mse: 0.28946685791015625

#shuffle=True => 성능이 더 좋아짐
loss: 0.01298578828573227
mse: 0.09316738694906235

#validation =0.2 => 성능이 좋아져야 하는데 성능떨어짐, 훈련량 자체가 떨어져서 성능이 떨어질 수도 있음
#loss=0.003
#mae=0.05

#

#R2,RMSE