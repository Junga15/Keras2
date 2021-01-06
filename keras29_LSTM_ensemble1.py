# #실습
#과제.LSTM 2개가 들어간 앙상블 모델을 구현하라.
#keras15_ensemble2.py(다대일앙상블 함수 LSTM모델)을 가져옴
#프레딕트 넣을 때 리쉐이프 해야함, x
# 로스, 
# 한개의아웃풋 지표 파악
# 데이터 구조파악
#y프레딕트값이 하나가 나와야함. 85 근사치값


#1.데이터 구성
import numpy as np

from numpy import array #이것도 상관없

x1=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

y=array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

print("x1.shape:",x1.shape) #(13,3)
print("x2.shape:",x2.shape) #(13,3)
print("y.shape:",y.shape) #(13,)

#.LSTM모델을 만들기 위한 ※keras23_LSTM1 #1.데이터 재구성 부분 참고 x=x.reshape(4,3,1) 
x1=x1.reshape(x1.shape[0],x1.shape[1],1) #리쉡할 때 x1.shape[0] 0번째 열 
x2=x2.reshape(x2.shape[0],x2.shape[1],1)
print(x1.shape) #(13, 3, 1)
print(x2.shape) #(13, 3, 1)

#2. 모델 구성 => keras15_ensemble2.py(다대일앙상블모델) 끌거옴,train_test_split과 함께

from tensorflow.keras.models import Model #함수형 모델이므로 import Sequential,Model이 아님
from tensorflow.keras.layers import LSTM,Dense,Input #명시법 잘봐두기!★ LSTM 함수형 모델
'''
15_2 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM => 내가 구성한 잘못된 표기
'''
#모델1
input1=Input(shape=(3,1)) #(13,3,1)을 인풋쉐이프로 넣어줄때 맨앞 날리고 (3,1)
dense1=LSTM(10,activation='relu')(input1) #이부분이 덴스 아닌 LSTM
dense1=Dense(10, activation='relu')(dense1)
#output1=Dense(5)(dense1) =>어짜피 변수표기는 내가 하는 것이므로 마지막에 output1은 내 선택사항

#모델2
input2=Input(shape=(3,1)) 
dense2=LSTM(10,activation='relu')(input2)
dense2=Dense(10, activation='relu')(dense2)

#모델병합concatenate: 사슬같이 잇다
from tensorflow.keras.layers import concatenate
merge1=concatenate([dense1,dense2])
#merge 합치다 첫번째 모델과 두번째 모델 중간중간 가중치 값 계산되서 마지막에 계산된 것 합치면 됨
#dense1,2줄 모두 layer, merge,middle 모두 레이어

#중간층 모델구성:생략가능
middle1=Dense(15)(merge1)
middle1=Dense(15)(middle1)

#모델 분기1
output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(1)(output1) #아웃풋 노드를 3개로 하면 프레딕트값이 3개 나옴, 1개해야 1개가 나옴

#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=output1)

#model.summary()

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit([x1,x2],y,epochs=10,verbose=1) #배치사이즈 넣으면 에러뜸,왜?

#4.평가, 예측

#.로스외의 평가지표 예측
result=model.evaluate([x1,x2],y)  
print(result)

#mse,mae [751.44140625, 16.33169174194336]

'''
#keras15_2
#   [1번째는 대표loss,2첫번재 모델loss, 3두번재모델loss, 4첫번째 모델metrics(mse), 5두번째 모델(mse)]    
#   1번째=2번째+3번째, 1번째=4번째+5번째   #metrics가 mae일 경우, 29,38이런식으로 나옴

#print("model.metrics_names:",model.metrics_names)
#model.metrics_names: ['loss', 'dense_12_loss', 'dense_16_loss', 'dense_12_mse', 'dense_16_mse']
'''
#y 프레딕트값 예측

#1단계.먼저 주어진 x1,x2_프레딕트값을 리쉐입을 해준다.
'''
#1.데이터 구성에서 주어진 값
x1_predict=array([55,65,75]) #(3,)=(1,3):Dense모델에서 사용->LSTM모델(3차원)인(1,3,1)로 리쉡
x2_predict=array([65,75,85]) #(3,)=(1,3):Dense모델에서 사용->LSTM모델(3차원)인(1,3,1)로 리쉡
'''
x1_pred=x1_predict.reshape(1,3,1) #=>(1,3,1)로 리쉡 해준다는 것
x2_pred=x2_predict.reshape(1,x2.shape[1],1)
#x2=x2.reshape(x2.shape[0],x1.shape[1],1) =>위의 x값 리쉡한 것 참조

print(x1_pred.shape) #(1, 3, 1)
print(x2_pred.shape) #(1, 3, 1)

#2.2단계.리쉡한 x1_프레딕트값,x2_프레딕트값을 통해 y프레딕트값을 구한다.
y_pred=model.predict([x1_pred,x2_pred]) #뒤에 ,y 써줘야함,위의 문장이 완성되려면
print(y_pred) #값 8.5 근사값 1개가 나와야 하는데 [[ 3.9641004 14.315914   9.244208 ]] 3개가 나옴

#keras15_2 y예측값
# y1_predict=model.predict([x1_predict,x2_predict])
# print("y1_predict: \n",y1_predict)