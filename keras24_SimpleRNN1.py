#심플RNN과 LSTM 성능비교하기: 심플알앤앤이 높기도하고 LSTM이 높기도 하다.
#심플RNN은 게이트가 없다.4가 없어짐 따라서 서머리했을 때 파라미터도 480개가 아닌 120개임(10번 한번순환하는 것은 똑같음,되돌림만 있는 상태)
#심플RNN도 Dense모델보다는 성능이 좋으나 오래걸림, 되돌림(역전파)가 있으므로
#문제점: 앞쪽의 연산이 뒷쪽에 연산에 영향을 잘 주지못함. 데이터가 많아질수록 LSTM에 밀림
#따라서 LSTM의 목적은 앞쪽의 데이터를 최대한 땡겨보자.결과값에 최대한 영향을 주게
#(GRU의 경우 리스트셀 게이트 한개를 뺌,가중치가 적다는 확실한 이점,그럼에도 LSTM과 성능은 거의 유사함)
#심플RNN의 디폴트 액티베이션: 탄젠트함수 

#1.데이터
import numpy as np

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4,3) 스칼라가 4개
y=np.array([4,5,6,7]) #(4,) y는 1,칼럼이1,아웃풋이 1=>벡터가 1

print("x.shape:",x.shape)
print("y.shape:",y.shape)

#1.2 데이터 재구성 =>reshape

x=x.reshape(4,3,1) 

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN
model=Sequential()
model.add(SimpleRNN(10,activation='relu',input_shape=(3,1))) #(3,1)에서 3은 timesteps,1은 input_dim, 여기의 디폴트 활성화함수가 탄젠트함수
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1)) #아웃풋노드의 갯수는 내가 정하는 것, 여기의 활성화함수 디폴트값의 리니어, 헷갈리지 말기!!

model.summary() 

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([5,6,7]) #(3,) =>1행
x_pred=x_pred.reshape(1,3,1) #1행 3열에 1개씩 자름, LSTM에 쓸수있는 구조로 바꿈, 데이터는 그대로임,소실없음

result=model.predict(x_pred)
print(result)

'''
심플RNN1 로스와 예측값
0.020261690020561218
[[8.278632]]
'''
