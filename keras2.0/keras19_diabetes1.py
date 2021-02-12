#18번1~6 당뇨병 모델에 적용해서 다시 연습
#하이퍼파라미터 튜닝 거의 안함

'''
#19_1 전처리 전 모델구성
loss,mae: [8060.69677734375, 69.75835418701172]
RMSE: 89.78138394961934
R2: -0.3997644530877227
'''

#0.넘파이 라이브러리 불러오기
import numpy as np

#1.1. 데이터 불러오기 및 데이터 구성
from sklearn.datasets import load_diabetes 
#사이킷런에서 제공하는 당뇨병 데이터셋

datasets=load_diabetes() #당뇨병 데이터셋 선언
x=datasets.data 
y=datasets.target 

#1.2.x와 y의 구조파악 (=> #2.모델구성시 사용하기 위해 구해야 함)
print(x.shape) #(442, 10)
print(y.shape) #(442,)

#1.3.트레인,테스트 분리하기
from sklearn.model_selection import train_test_split 
#모델바셀렉션에 트레인테스트스플릿 있음
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#트레인테스트스플릿(x,y,...) 명시 가능함

#2.모델구성: 시퀀셜모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=10))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x_train,y_train,validation_split=0.2,
          epochs=100,batch_size=10)

#4.훈련,평가

#4.1.로스값 구하기
loss=model.evaluate(x_test,y_test,
                    batch_size=10)
print('loss:',loss)

#4.2.RMSE값과 R2값 구하기

#1)y_predict값 구하기
y_predict=model.predict(x_test)
#from sklearn.metrics import mean_squared_error,r2_score

#2)RMSE 구하기:def함수를 이용해서 RMSE 선언 후 RMSE값 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE:",RMSE(y_test,y_predict))

#3)R2 구하기
from sklearn.metrics import r2_score
R2=r2_score(y_test,y_predict)
print("R2:",R2)



