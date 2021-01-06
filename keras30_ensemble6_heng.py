#행이 다른 앙상블 모델에 대해 공부 
# =>오류뜸, 기본모델 뿐만 아니라 앙상블모델도 행을 맞춰줘야 한다.

'''
ValueError: Data cardinality is ambiguous:
  x sizes: 10, 13
  y sizes: 10, 13
Please provide data which shares the same first dimension.
'''
#1.데이터 구성
import numpy as np

from numpy import array #이것도 상관없

x1=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]]) #(10,3)
x2=array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
         #(13,3)
y1=array([4,5,6,7,8,9,10,11,12,13]) 
y2=array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict=array([55,65,75])
x2_predict=array([65,75,85])

print("x1.shape:",x1.shape) #(10, 3)
print("x2.shape:",x2.shape) #(13, 3)
print("y1.shape:",y1.shape) #(10,)
print("y2.shape:",y2.shape) #(13,)

#2. 모델 구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

#모델1
input1=Input(shape=(3,)) 
dense1=Dense(10,activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)
output1=Dense(3)(dense1) 

#모델2
input2=Input(shape=(3,)) 
dense2=Dense(10,activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
output2=Dense(3)(dense2)

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
output1=Dense(3)(output1)

#모델 분기2
output2=Dense(30)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(3)(output2)

#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=[output1,output2])

#model.summary()

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit([x1,x2],[y1,y2],epochs=10,verbose=1) #배치사이즈 넣으면 에러뜸,왜?

#4.평가, 예측

#.로스외의 평가지표 예측
result=model.evaluate([x1,x2],[y1,y2])  
print(result)


#2.2단계.리쉡한 x1_프레딕트값,x2_프레딕트값을 통해 y프레딕트값을 구한다.
y_pred=model.predict([x1_pred,x2_pred],[y1_pred,y2_pred]) #뒤에 ,y 써줘야함,위의 문장이 완성되려면
print(y_pred) #값 8.5 근사값 1개가 나와야 하는데 [[ 3.9641004 14.315914   9.244208 ]] 3개가 나옴

#keras15_2 y예측값
# y1_predict=model.predict([x1_predict,x2_predict])
# print("y1_predict: \n",y1_predict)




