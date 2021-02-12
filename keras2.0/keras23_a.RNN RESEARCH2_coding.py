#※참고자료: [케라스] 무작정 튜토리얼 11 - LSTM(feat. RNN) 구현하기 :: ebb and flow


from numpy import array 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM 

#1.데이터 구성
x=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])

y=array([4,5,6,7]) #(1,4)=>(4,)로 표시
print(y.shape) #(4,)

# x   y 
# 123 4
# 234 5
# 345 6
# 456 7

print(x) #[[1 2 3][2 3 4][3 4 5][4 5 6]]
print(x.shape) #(4,3)
print('-----x.reshape-----')
x=x.reshape(x.shape[0],x.shape[1],1) 
print(x.shape) #(4, 3, 1)=>리쉡후의 x모양: 전체곱수가 이전 모양의 곱수와 같아야 4x3=4x3x1
print(x) #[[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]]]


#2.모델 구성
model=Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1)))
#Dense와 사용법이 동일하나 input_shape=(열,몇개씩 잘라작업)
# : 여기서는 위의 마지막 x출력값처럼 3개의 열을 1개씩 잘라서 작업하라는 명령
model.add(Dense(5))
model.add(Dense(1))

model.summary() #서머리 함수

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 541

# LSTM의 파라미터를 계산하는 법은 Dense레이어와 다릅니다.
# 4(LSTM의 4개의 게이트)X([인풋노드3+바이어스1+아웃풋노드5])X5(한바퀴돌기 때문에 아웃풋노드,역전파의 개념과 유사함)=480개
# =[(4X10+4X1+4X5)]X5도 가능=>[]LSTM의 상층게이트
#※


