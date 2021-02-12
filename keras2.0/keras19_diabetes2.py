

'''
#19_2 수동전처리 모델구성 =>약간 이상하고 잘못된 방식
트레인테스트분리후 x=(x-np.min(x))/(np.max(x)-np.min(x)) 넣어줌

loss :  15457.830078125
RMSE: 124.32952179395143
r2: -462.74511818484143
'''

#0.넘파이 라이브러리 불러오기
import numpy as np

#1.데이터
#1)데이터 불러오기 및 데이터 구성
from sklearn.datasets import load_diabetes

datasets=load_diabetes()
x=datasets.data 
y=datasets.target 

#2)x와 y의 구조파악 (#2.모델구성시 사용)
print(x.shape) #(442, 10)
print(y.shape) #(442,)

#3)트레인,테스트 분리하기
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8)
#4)트레인테스트 분리 후 수기로 민맥스스칼라(전처리) 진행
x=(x-np.min(x))/(np.max(x)-np.min(x))                                                           
'''
print(x_train) #값
print(x_train.shape) #(353, 10) 행값:353.6=442x0.8 
print(x_test) #값
print(x_test.shape) #(89, 10) 행값:89=442-353
print(y_train)  # [ 78.  71.  44....90. 265. 237.]
print(y_train.shape) #(353,)
print(y_test) #값 [163. 141. 185....74. 139. 258.]
print(y_test.shape) #(89,)
'''

#2.모델구성
from tensorflow.keras.models import Sequential 
#함수를 불러올때는 텐서플로우,케라스에서 '모델스'로 불러온다.
from tensorflow.keras.layers import Dense 
#층을 불러올때는 텐서플로우,케라스에서 '레이어스'로 불러온다.

model=Sequential()
#model.add(Dense(10,input_dim=10,activation='relu'))
model.add(Dense(10,input_shape=(10,),activation='relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,
         epochs=100,batch_size=10,
         validation_split=0.2)

#4.평가,예측:모델에 대한 평가,예측을 여러가지 지표로 확인

#1) 로스값 구하기
loss=model.evaluate(x_test,y_test,batch_size=10)
print('loss : ',loss)

#2) y예측값 구하기
y_predict = model.predict(x_test) #테스트한 x값을 통해 y값을 예측함

#3) y예측값을 통한 RMSE와 2R2 구하기

#RMSE(루트평균제곱오차): def함수를 통해 RMSE 선언 후 RMSE값 구하기
from sklearn.metrics import mean_squared_error #사이킷런 중 메트릭스에 있음

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE',RMSE(y_test,y_predict))
#※"RMSE:"는 틀린문법, "RMSE"가 맞는 문법임 

#R2
from sklearn.metrics import r2_score #r2_score에 있음
r2=r2_score(y_predict,y_test) #r2_score()
print("r2:",r2)