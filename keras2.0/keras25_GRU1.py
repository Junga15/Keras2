#실습. keras23_LSTM1로 GRU 코드분석 (※심플알앤앤1참고)
#GRU 파라미터 분석할 것 =>파라미터 390개 나옴,
#예측답(오늘의 과제): 아웃풋이 2개 게이트가 3개? => LSTM에서 망각을 뺌?
#답?: 3(LSTM의 3개의 게이트)X([인풋노드1+바이어스1+아웃풋노드10])X10(한바퀴돔,역전파의 개념과 유사함)=390개
#성능은 LSTM과 유사함
#결과치(로스와 프레딕트값)을 LSTM과 비교
'''
GRU1의 로스값과 프레딕트값
0.0069593098014593124
[[7.917649]]
'''

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
from tensorflow.keras.layers import Dense,GRU
model=Sequential()
model.add(GRU(10,activation='relu',input_shape=(3,1))) #(3,1)에서 3은 timesteps,1은 input_dim, 여기의 디폴트 활성화함수가 탄젠트함수
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