#실습,드롭아웃 적용
#21.1 가져옴 #keras21_1_breastcancer1,py

'''
[드롭아웃 적용 전]
loss: 0.07230956852436066
acc,mae: 0.9780701994895935 0.053853537887334824

[드롭아웃 적용 후] =>거의 비슷하게 나옴
loss: 0.07619798928499222
acc,mae: 0.9736841917037964 0.053503137081861496
'''

#0.넘파이 라이브러리 가져오기
import numpy as np

#1.데이터셋 불러오기
from sklearn.datasets import load_breast_cancer

datasets=load_breast_cancer()
'''
print(datasets.DESCR)
print(datasets.feature_names) #교육용이라 확인가능함
'''
x=datasets.data #실제 업무에서 데이터셋을 분석해야 모델링이 가능함
y=datasets.target

print(x.shape) #이거 확인하는 게 가장 중요 #칼럼이 30개, (569, 30)
print(y.shape) #(569,) =>벡터(1차원텐서)가 1이고 스칼라(0차원텐서)가 569개
print(x[:5]) #x안에 있는 데이터 5개만 보기=
print(y) #아웃풋이 1개의 칼럼 #회귀문제는 실제값을 출력하는 것이고 분류값은 0,1로 분류하는 것


#1.데이터 전처리 train test split,minmax하기
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

print(x_test.shape) #(456, 30)
print(y_test.shape) #(456,)

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(np.max(x),np.min(x)) #395: 해당칼럼의 최대가 395


#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(30,))) #액티베이션 처음과 중간은 다른거 넣어도 마지막엔 시그모이드 함수 넣어야함
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu')) 
model.add(Dropout(0.2))                                    #이 레이어만 있는 경우도 상관없음, 히든이 없는 모델도 가능함
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))    
model.add(Dropout(0.2))                                                                        
model.add(Dense(1,activation='sigmoid')) #액티베이션이 디폴트값은 리니어,따라서 범위는 무한대~무한대, 나오는 값은
 #시그모이드함수는 0과 1사이에 한정을 짓는 함수 0<n<1, 어떤 레이어도 0과 1사이에 수렴한다.
 #렐루함수는 액티베이션에 대해서 가중치 연산을 제어한다, 값이 음수일 때 상쇄, 0이상의 값만 전달해줌.
                                         #렐루함수는 0보다 같거나 크거나 =>렐루가 성능이 좋다, 평타 85%이상침(시그모이드는 그렇지 않음)
                                         #max(0,x)
                                         #최초에 나온 모델은 0과 1사이에 있는 시그모이드 함수

#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc','mae']) #100% 바이너리 크로센트로피, 메트릭스는 거의 에큐러시 넣음
                                                                           #회귀모델일 때는 로스가 mse,mae 이진분류일 때는 로스가 바이너리 크로센트로피
                                                                           #처음에 로스를 mse를 썼기 때문에 로스가 1000~이상,애큐레시가 0.4 모델링을 잘못한것임
#model.compile(loss='mean_squred_error',optimizer='adam',metrics=['acc','mae'])
model.fit(x,y,epochs=10,validation_split=0.2)                              
loss=model.evaluate(x,y)

#4.평가,예측
#print(loss) #[0.45258259773254395, 0.9068541526794434]

loss,acc,mae = model.evaluate(x_test,y_test,batch_size=1)
print('loss:',loss)
print('acc,mae:',acc,mae)

'''
loss: 0.07230956852436066
acc,mae: 0.9780701994895935 0.053853537887334824
'''

'''
y_predict=model.predict(x_test) 
print(y_predict)
# '''
# print(x[-5:-1])
# y_pred=model.predict(x[-5:-1])
# #print(y_test)
# print(y_pred) 
# #[[2.9731094e-05]
# #  [3.3152287e-04]
# #  [8.9458991e-03]
# #  [6.9059215e-06]]

# y_pred = model.predict(x[-5:-1])
# print(np.argmax(y_pred,axis=-1)) #[0 0 0 0] axis값이 -1인가,0인가,1인가? =>argmax.py참고
# print(y[-5:-1]) #[0 0 0 0]

# #argmax
# # Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   
# # if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).*
# #  `(model.predict(x) > 0.5).astype("int32")`,   
# # if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).

