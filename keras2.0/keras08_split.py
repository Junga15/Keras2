from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#np.array()
#array()

#1.데이터
x=np.array(range(1,101))  #x=np.array(range(100))
y=np.array(range(101,201))

x_train=x[:60] #값:1~60 => 순서 0번째부터 59번째까지 값: 1~60
x_val=x[60:80] #값:61~80
x_test=x[80:100] #값:81~100
#list slicing 리스트 슬라이싱

'''
y_train = y[:60] #값 1~60 => 순서0번째부터 59번째까지 값 1~60
y_val = y[60:80] #값 61~80
y_test = y[80:100] #값 81~100
#리스트 슬라이싱
'''
y_train=y[:60]
y_val=y[60:80]
y_test=y[80:100]

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))

#4.평가,예측
results = model.evaluate(x_test, y_test, batch_size=1)
print('results : ', results)

y_predict=model.predict(x_test) #x_pred를 넣어서 예측한 값
#print('y_predict:',y_predict)

#사이킷런이 뭔지 2분 검색 sikit-learn
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) #squrt는 루트
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

#R2 -> 0.99999 
#model.add(Dense(5))x5,epochs=1000