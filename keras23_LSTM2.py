#keras23_LSTM2.py 실습

#input_shape / input_length / input_dim
#이페이지 바꾼 건 model.add(LSTM(10,activation='relu',input_length=3,input_dim=1)) #위의 것 이렇게 실행가능, 인풋딤은 피쳐, 인풋렝스는 타임스텝스
#이거밖에 없음

#1.데이터
import numpy as np

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4,3) 스칼라가 4개
y=np.array([4,5,6,7]) #(4,) y는 1,칼럼이1,아웃풋이 1=>벡터가 1

print("x.shape:",x.shape)
print("y.shape:",y.shape)

x=x.reshape(4,3,1) 

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model=Sequential()
#model.add(LSTM(10,activation='relu',input_shape=(3,1))) 
model.add(LSTM(10,activation='relu',input_length=3,input_dim=1)) #위의 것 이렇게 실행가능, 인풋딤은 피쳐, 인풋렝스는 타임스텝스
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1)) #아웃풋노드의 갯수는 내가 정하는 것, 여기의 활성화함수 디폴트값의 리니어, 헷갈리지 말기!!

model.summary() #질문:Param#이 왜 480인가? =>결과값 나올때까지 오래걸림
                #답: 4(LSTM의 4개의 게이트)X([인풋노드1+바이어스1+아웃풋노드10])X10(한바퀴돔,역전파의 개념과 유사함)=480개
                #   =[(4X1+4X1+4X10)]X10도 가능=>[]LSTM의 상층게이트

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
