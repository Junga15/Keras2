#실습: 19_1번부터 5번,얼리스타핑까지18번 적용
#먼저 1번째것을 최적화로 만들고 그 다음것부터 성능비교
#총 6개의 파일을 완성하시오.

'''
1.전처리전
loss,mae: 3440.106689453125 46.08512878417969
RMSE: 58.65241827408525
R2: 0.43783686862029003
'''

import numpy as np 
from sklearn.datasets import load_diabetes

dataset= load_diabetes()
x=dataset.data
y=dataset.target

print(x[:5])
print(x[:10])

# print(np.max(x),np.min(y)) #0.198787989657293 25.0
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape,y.shape) #(442,10) (442,) 인풋쉐이프 10?, 인풋딤 10


#1.데이터 구성 및 전처리:train_test_split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=False,test_size=0.8,random_state=66)

print(x_test.shape) #(354, 10)
print(y_test.shape) #(354,)

#x=(x-np.min(x))/(np.max(x)-np.min(x))
#print(np.max(x[0]))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
print(np.max(x),np.min(x)) #1.0 0.0



'''
#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
#model.add(Dense(128,activation='relu',input_dim=10))
#model.add(Dense(128,activation='relu',input_shape=10))=>인풋쉐이프?
model.add(Dense(5,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,epochs=150,batch_size=1,validation_split=0.5)

#4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=1)
print('loss,mae:',loss,mae)

#5.RMSE와 R2값 구하기
y_predict=model.predict(x_test)
#print=y_predict

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)
'''