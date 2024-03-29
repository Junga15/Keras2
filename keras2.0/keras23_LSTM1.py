<<<<<<< HEAD:keras2.0/keras23_LSTM1.py
#keras23_LSTM1.py 실습

#1.데이터
import numpy as np

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4,3) 스칼라가 4개
y=np.array([4,5,6,7]) #(4,) y는 1,칼럼이1,아웃풋이 1=>벡터가 1

print("x.shape:",x.shape)  #(4,3)
print("y.shape:",y.shape)  #(4,)

#1.2 데이터 재구성 =>reshape
#한개씩 잘라서 작업하기 위해 쉐이프를 바꾼다. LSTM레이어를 쓰려면 3차원 데이터야함
# ※Dense는 데이터 2차원(인풋딤은 행무시하고 1차원),LSTM(RNN계열,시계열데이터)은 3차원(인풋딤은 행무시하고 2차원),CNN은 4차원
#LSTM이라고 하면 RNN으로 알아들으면 됨,시계열 데이터->자연어처리,챗봇 시용
#LSTM(시계열 데이터)은 y값이 없다.=>3개의 칼럼으로 y값을 예측함 (ex.삼성전자 주가예측), 우리가 y데이터를 만드는 것(아예 없는 것은 아님)
x=x.reshape(4,3,1) #[[[1],[2],[3]],[[2],[3],[4]]...] #원소는 똑같음, 데이터 자체의 손실은 없음,데이터의 갯수가 변화되면 안됨
                                                     #총 곱한 값은 같아야 함 ex>(5,4),(5,4,1),(5,2,2) =>예를 들어 2개씩 자른다면 열은 2가 되어야함
#-1은 행렬이든 넘파이배열이든,판다스 배열이든 제일 끝에 있는 숫자를 가리킴

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1))) #(3,1)에서 3은 timesteps,1은 input_dim, 여기의 디폴트 활성화함수가 탄젠트함수
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

'''
0.003668485675007105
[[8.087947]]
'''
=======
#keras23_LSTM1.py 실습

#1.데이터
import numpy as np

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4,3) 스칼라가 4개
y=np.array([4,5,6,7]) #(4,) y는 1,칼럼이1,아웃풋이 1=>벡터가 1

print("x.shape:",x.shape)  #(4,3)
print("y.shape:",y.shape)  #(4,)

#1.2 데이터 재구성 =>reshape
#한개씩 잘라서 작업하기 위해 쉐이프를 바꾼다. LSTM레이어를 쓰려면 3차원 데이터야함
# ※Dense는 데이터 2차원(인풋딤은 행무시하고 1차원),LSTM(RNN계열,시계열데이터)은 3차원(인풋딤은 행무시하고 2차원),CNN은 4차원
#LSTM이라고 하면 RNN으로 알아들으면 됨,시계열 데이터->자연어처리,챗봇 시용
#LSTM(시계열 데이터)은 y값이 없다.=>3개의 칼럼으로 y값을 예측함 (ex.삼성전자 주가예측), 우리가 y데이터를 만드는 것(아예 없는 것은 아님)
x=x.reshape(4,3,1) #[[[1],[2],[3]],[[2],[3],[4]]...] #원소는 똑같음, 데이터 자체의 손실은 없음,데이터의 갯수가 변화되면 안됨
                                                     #총 곱한 값은 같아야 함 ex>(5,4),(5,4,1),(5,2,2) =>예를 들어 2개씩 자른다면 열은 2가 되어야함
#-1은 행렬이든 넘파이배열이든,판다스 배열이든 제일 끝에 있는 숫자를 가리킴

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1))) #(3,1)에서 3은 timesteps,1은 input_dim, 여기의 디폴트 활성화함수가 탄젠트함수
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

'''
0.003668485675007105
[[8.087947]]
'''
>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras23_LSTM1.py
