<<<<<<< HEAD:keras2.0/keras15_ensemble5.py
#10_3 데이터끌어온것임 #다대다 앙상블
#실습.위 두개의 모델을 퉁칠 수 있는 RMSE와 R2값을 구하라.(위의 print(loss)=두개통합 로스값처럼)
#=> 앙상블 모델의 RMSE,R2

import numpy as np

x1=np.array([range(100),range(301,401),range(1,101)]) #print(x1.shape)는 (3,100)
y1=np.array([range(711,811),range(1,101),range(201,301)])


x2=np.array([range(101,201),range(411,511),range(100,200)])
y2=np.array([range(501,601),range(711,811),range(100)])

x1=np.transpose(x1) #4개 모두 (100,3)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,shuffle=False,train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,shuffle=False,train_size=0.8)

#2. 모델구성
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

#모델병합 / concatenate: 사슬같이 잇다
from tensorflow.keras.layers import concatenate, Concatenate
#소문자는 텐서플로우 메서드이고 대분자는 케라스 클래스임
#from keras.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate 
merge1=concatenate([dense1,dense2]) 
#merge 합치다 첫번째 모델과 두번째 모델 중간중간 가중치 값 계산되서 마지막에 계산된 것 합치면 됨
#dense1,2줄 모두 layer, middle1,output2 라인 모두 가각의 레이어이며 모델은 총 5개로 구성
middle1=Dense(30)(merge1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

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

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae','mse'])
model.metrics_names()
model.fit([x1_train,x2_train],[y1_train,y2_train],
          epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가, 예측
loss=model.evaluate([x1_test,x2_test],
                    [y1_test,y2_test], batch_size=1)

print(loss)
#답 [3462.90771484375, 2320.569580078125, 1142.33837890625, 2320.569580078125, 1142.33837890625]
#   [1번째는 대표loss,2첫번재 모델loss, 3두번재모델loss, 4첫번째 모델metrics(mse), 5두번째 모델(mse)]    
#   1번째=2번째+3번째, 1번째=4번째+5번째   #metrics가 mae일 경우, 29,38이런식으로 나옴
#   로스값은 상대치임

print("model.metrics_names:",model.metrics_names)
#model.metrics_names: ['loss', 'dense_12_loss', 'dense_16_loss', 'dense_12_mse', 'dense_16_mse']
#실습.위 두개의 모델을 퉁칠 수 있는 RMSE와 R2값을 구하라.(위의 print(loss)=두개통합 로스값처럼)
#=> 앙상블 모델의 RMSE,R2

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
#print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_p
# redict,y_test))

RMSE1=RMSE(y1_test,y1_predict)
RMSE2=RMSE(y2_test,y2_predict)
RMSE=(RMSE1+RMSE2)/2
print("RMSE1:",RMSE1)
print("RMSE2:",RMSE2)
print("RMSE:",RMSE)
'''
#R2구하기
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

y_pred2=model.predict(x_pred2) 
print(y_pred2) 
'''
from sklearn.metrics import r2_score
r2_1=r2_score(y1_test,y1_predict)
r2_2=r2_score(y2_test,y2_predict)
r2=(r2_1+r2_2)/2

print("r2_1:",r2_1)
print("r2_2:",r2_2)
print("r2:",r2)
=======
#10_3 데이터끌어온것임 #다대다 앙상블
#실습.위 두개의 모델을 퉁칠 수 있는 RMSE와 R2값을 구하라.(위의 print(loss)=두개통합 로스값처럼)
#=> 앙상블 모델의 RMSE,R2

import numpy as np

x1=np.array([range(100),range(301,401),range(1,101)]) #print(x1.shape)는 (3,100)
y1=np.array([range(711,811),range(1,101),range(201,301)])


x2=np.array([range(101,201),range(411,511),range(100,200)])
y2=np.array([range(501,601),range(711,811),range(100)])

x1=np.transpose(x1) #4개 모두 (100,3)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,shuffle=False,train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,shuffle=False,train_size=0.8)


#2. 모델구성
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

#모델병합 / concatenate: 사슬같이 잇다
from tensorflow.keras.layers import concatenate, Concatenate
#소문자는 텐서플로우 메서드이고 대분자는 케라스 클래스임
#from keras.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate 
merge1=concatenate([dense1,dense2]) 
#merge 합치다 첫번째 모델과 두번째 모델 중간중간 가중치 값 계산되서 마지막에 계산된 것 합치면 됨
#dense1,2줄 모두 layer, middle1,output2 라인 모두 가각의 레이어이며 모델은 총 5개로 구성
middle1=Dense(30)(merge1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

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

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae','mse'])
model.metrics_names()
model.fit([x1_train,x2_train],[y1_train,y2_train],
          epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가, 예측
loss=model.evaluate([x1_test,x2_test],
                    [y1_test,y2_test], batch_size=1)

print(loss)
#답 [3462.90771484375, 2320.569580078125, 1142.33837890625, 2320.569580078125, 1142.33837890625]
#   [1번째는 대표loss,2첫번재 모델loss, 3두번재모델loss, 4첫번째 모델metrics(mse), 5두번째 모델(mse)]    
#   1번째=2번째+3번째, 1번째=4번째+5번째   #metrics가 mae일 경우, 29,38이런식으로 나옴
#   로스값은 상대치임

print("model.metrics_names:",model.metrics_names)
#model.metrics_names: ['loss', 'dense_12_loss', 'dense_16_loss', 'dense_12_mse', 'dense_16_mse']
#실습.위 두개의 모델을 퉁칠 수 있는 RMSE와 R2값을 구하라.(위의 print(loss)=두개통합 로스값처럼)
#=> 앙상블 모델의 RMSE,R2

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
#print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_p
# redict,y_test))

RMSE1=RMSE(y1_test,y1_predict)
RMSE2=RMSE(y2_test,y2_predict)
RMSE=(RMSE1+RMSE2)/2
print("RMSE1:",RMSE1)
print("RMSE2:",RMSE2)
print("RMSE:",RMSE)
'''
#R2구하기
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

y_pred2=model.predict(x_pred2) 
print(y_pred2) 
'''
from sklearn.metrics import r2_score
r2_1=r2_score(y1_test,y1_predict)
r2_2=r2_score(y2_test,y2_predict)
r2=(r2_1+r2_2)/2

print("r2_1:",r2_1)
print("r2_2:",r2_2)
print("r2:",r2)
>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras15_ensemble5.py
