#keras28_LSTM.py 실습
#과제1.keras23_3을 카피해서 LSTM층을 2개를 만드세요.
#예)
#model.add(LSTM,input_shape=(3,1)))
#model.add(LSTM(10))
#이때 오류생기는 것을 return_sequences=True로 잡아줘야 한다. => 어디 넣어야? LSTM input층에
#과제2.LSTM층 1개와 2개쓴 것이상(3,4개,10개이상도 가능) 성능차이 비교
#답2. 2개 이상 달 경우 성능이 좋지 않다. 

import numpy as np

#1.데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape:",x.shape) #(13,3)
print("y.shape:",y.shape) #(3,)

#1.2 데이터 재구성 =>reshape
x=x.reshape(13,3,1) #13개의 행에 3개의 칼럼을 가진 행렬을 1개씩 잘라서 훈련시킴, 기계가 받아들이는 것은 (0[none],3,1)로 받아들임
x=x.reshape(x.shape[0],x.shape[1],1) 
print(x.shape[0]) #13
print(x.shape[1]) #3

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1),return_sequences=True)) #인풋쉐이프는 (None,3,1)을 none을 제외하고 (3,1)명시
#모델 서머리 넌컴파3,1로 들어가서 넌컴마,3,1,10(인풋노드의 갯수 맨뒤에 붙음)으로 던져줌=>파라미터 840개로 확늠
#리턴시퀀스 쓰면 2개로 받아들인다.

model.add(LSTM(10))                                                           #엘에스티엠은 3차원인데 나오는 것은 2차원으로 빼줌                                                  
model.add(Dense(10)) #덴스가 던져주는 것 대부분 (0none,10)                     #엘에스티엠에서 다음 형식으로 던져줄때 이전 차원을 던져줌
model.add(Dense(10))
model.add(Dense(1))

model.summary()
#리턴ㅅ
#ValueError: Input 0 of layer lstm_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 10]

'''
#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([50,60,70]) #(3,) =>1행
x_pred=x_pred.reshape(1,3,1) 

result=model.predict(x_pred)
print(result)

'''
# 0.05104106664657593
# [[80.62689]]
# #몇번 돌려도 값이 비슷해야 좋은 모델임,실습을 많이 해봐야 감이늠,실습을 많이 해봐야 하는 이유
'''
'''