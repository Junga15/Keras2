#다중분류,y에 대한 원핫인코딩
#실습,과제
#1.프레딕트값이 소수점에서 원하는 값(이진분류0,1)으로 나오는 코드로 변경하기=>if함수나 argmax 사용
#2.y에 대한 원핫인코딩 tensorflow/keras용 to_categorical을 sklearn용으로 바꾸기.
#2개의 모델 얼리스탑핑 적용하기

import numpy as np
from sklearn.datasets import load_iris

#1.데이터
#x,y=load_iris(return_X_y=True)

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(dataset.DESCR) 
# print(dataset.feature_names) #열,피쳐,칼럼,어트리뷰트가 4개

print(x.shape)  #(150,4)
print(y.shape)  #(150,)
print(x[:5]) #X는 컬럼별로 답이 있고 전처리가 안되어 있는 느낌?
print(y) #0,1,2가 50개씩 3종류로 있음 

#1.2.데이터 전처리
'''
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
y=to_categorical(y)
'''
#싸이킷런에서 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder()
onehot.fit(y.reshape(-1,1))
y=onehot.transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66)

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
print(np.max(x),np.min(x)) 

#원핫인코딩:y값을 0과 1의 값으로 분리하기=>y를 벡터화(0,1)하는 것을 말함 =>x,y스플릿전에 해야 에러가 안뜸,맨처음에 해주기
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
#y=to_categorical(y)
#y_train=to_categorical(y_train)
#y_test=to_categorical(y_test)
print(y) #[1. 0. 0.], [0. 1. 0.], [0. 0. 1.] 각각 50개씩
print(x.shape) #(150,4)
print(y.shape) #(150,3)


#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10,activation='relu',input_shape=(4,)))   
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10)) #loss: [0.5280014872550964, 0.8666666746139526, 0.09369666129350662]
model.add(Dense(3,activation='softmax')) #<다중분류 활성화 함수: 소프트맥스> 
                                         #소프트맥스 함수에 인풋값을 넣으면, 그 값들을 모두 0과 1사이의 값으로 정규화해주는데, 
                                         #이는 각 확률이 마이너스 값을 가지지 않고 더했을 때 총합이 1이 되는 것과 매우 흡사합니다.
                                         #다중분류일 경우에는 분류하고자 하는 종류의 갯수(결과값의 수,아웃풋수,원핫인코드한 쉐이프의갯수)를 
                                         # 마지막 노드의 숫자로 지정 =>여기서는 3개
                                         # 이3개의 노드의 합은 1이다.
                                         #이때 가장 큰 값이 매치가 된다. ex>0.49,0.31,0.2 일때 0.49가 채택됨 =>0.49가 1,0,0이됨
                                                                        #ex>0.1,0.2,0.7 일때 0.7이 채택됨 => 
                                         #원핫인코딩: #1.데이터전처리에서 y값 변환
                                         # : y값이 A,B,C가 0,1,2로 표기됐을 때 C=Bx2가 아니기 때문에
                                         #A(00 iris)는 1,0,0 B는 0,1,0 C는 0,0,1로 표기해서 변환시켜줌
                                    
                                         #y값 원핫인코딩 해줘야함
                                         #to_categorical(텐서플로)은 0부터 시작해야함=>텐서플로와 사이킷런의 차이점?=>나중에~
                                    

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc','mae'])
model.fit(x,y,epochs=100,validation_split=0.2)                              

#4.평가,예측
#print(loss) #[1.1920930376163597e-07, 0.3333333432674408, 0.8222556710243225] =>시그모이드 넣었을때 즉,잘못된 모델링
loss=model.evaluate(x_test,y_test,batch_size=20) #loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print(x[-5:-1])
y_pred=model.predict(x[-5:-1])
#print(y_test)
print(y_pred)
print(y[-5:-1])

