#x전체를 한번에 minmaxscalar이용하여 전처리한 후
#트레인테스트 분리후 x전체 민맥스스칼라전처리 진행함
'''
<#트레인테스트 분리후 x전체 민맥스스칼라전처리> =>하이퍼 파라미터 튜닝 안함
loss :  12458.0771484375
RMSE 111.61575900292273
r2: -197.97435089880764
'''

#0.넘파이 라이브러리 불러오기
import numpy as np

#1.데이터

#1)데이터 불러오기 및 데이터 구성
from sklearn.datasets import load_diabetes

datasets=load_diabetes()
x=datasets.data
y=datasets.target 

#2)x와 y의 구조파악: #2.에서 모델구성시 사용
print(x.shape) #(442, 10)
print(y.shape) #(442,)

#3)트레인,테스트 분리하기
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

print(x_test.shape) #(89,10)
print(y_test.shape) #(89,)

#4)민맥스 스칼라로 x전체 전처리 실행
from sklearn.preprocessing import MinMaxScaler
#사이킷런에 전처리(preprocessing)에서 민맥스스칼라 있음
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

print(x.shape)  # (442, 10)


#2.모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) 

'''
model.add=Dense로 정의함
Traceback (most recent call last):
    return np.sqrt(mean_squared_error(y_test,y_predict))
    r2=r2_score(y_test,y_predict) #r2_score()

ValueError: y_true and y_pred have different number of output (1!=10)

귀하의 y_test데이터 모양 (N, 1)이지만 출력 층에 10 개 뉴런을 넣어 때문에, 모델 오류 10 개 가지 예측을합니다.
출력 레이어의 뉴런 수를 1로 변경하거나 뉴런이 1 개만있는 새 출력 레이어를 추가해야합니다.

'''

#3.컴파일과 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, validation_split=0.2,
          epochs=10, batch_size=10)


#4.평가,예측:모델에 대한 평가,예측을 여러가지 지표로 확인

#1) 로스값 구하기
loss=model.evaluate(x_test, y_test, batch_size=10)
print('loss : ',loss)


print(x_test.shape)     # (89, 10)
print(y_test.shape)     # (89,)
#2) y예측값 구하기
y_predict = model.predict(x_test) #테스트한 x값을 통해 y값을 예측함
# print(y_predict)
print(y_predict.shape) # (89, 1) =>(89,)와 다릉
y = y_predict.reshape(89,) #(89,)
print(y_predict.shape) # (89, 1)

#3) y예측값을 통한 RMSE와 2R2 구하기
#RMSE(루트평균제곱오차): def함수를 통해 RMSE 선언 후 RMSE값 구하기
from sklearn.metrics import mean_squared_error #사이킷런 중 메트릭스에 있음
def rmse(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('rmse:',rmse(y_test,y_predict))

#R2
from sklearn.metrics import r2_score #r2_score에 있음
r2=r2_score(y_test,y_predict) #r2_score()
print("r2:",r2)
