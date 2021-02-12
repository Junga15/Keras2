<<<<<<< HEAD:keras2.0/keras15_ensemble3.py
#실습 
# 1.다:다 앙상블을 구현하시오. =>여기서는 2:3, y3추가
'''
RMSE: 43.0111289572088
r2: -54.637810952590826
'''
# 2.predict 구하기

import numpy as np

x1=np.array([range(100),range(301,401),range(1,101)]) #100개의 데이터 3개 - 100헹 3열 input_shape(3,)
x2=np.array([range(101,201),range(411,511),range(100,200)]) #output 3

y1=np.array([range(711,811),range(1,101),range(201,301)])
y2=np.array([range(501,601),range(711,811),range(100)])
y3=np.array([range(601,701),range(811,911),range(1100,1200)])

x1=np.transpose(x1) #아래 5개 모두(100,3)
x2=np.transpose(x2)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,shuffle=False,train_size=0.8)
x2_train,x2_test,y2_train,y2_test,y3_train,y3_test=train_test_split(x2,y2,y3,shuffle=False,train_size=0.8)

#2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,concatenate,Concatenate

#모델1
input1=Input(shape=(3,)) #input layer 구성
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
merge1=concatenate([dense1,dense2]) 
#merge 합치다 첫번째 모델과 두번째 모델 중간중간 가중치 값 계산되서 마지막에 계산된 것 합치면 됨
#dense1,2줄 모두 layer, merge,middle 모두 레이어
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

#모델 분기3
output3=Dense(30)(middle1)
output3=Dense(7)(output3)
output3=Dense(7)(output3)
output3=Dense(3)(output3)


#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=[output1,output2,output3])

model.summary()
'''
#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train],[y1_train,y2_train,y3_train],
          epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가, 예측
loss=model.evaluate([x1_test,x2_test],

model.summary()
print(loss)

#답 [3462.90771484375, 2320.569580078125, 1142.33837890625, 2320.569580078125, 1142.33837890625]
#   [1번째는 대표loss,2첫번재 모델loss, 3두번재모델loss, 4첫번째 모델metrics(mse), 5두번째 모델(mse)]    
#   1번째=2번째+3번째, 1번째=4번째+5번째   #metrics가 mae일 경우, 29,38이런식으로 나옴

y1_pred,y2_pred,y3_pred=model.predict([x1_test,x2_test])
print(y1_pred,y2_pred,y3_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y1_test,y2_test,y3_test,y1_pred,y2_pred,y3_pred):
    return np.sqrt((mean_squared_error(y1_test,y1_pred)+mean_squared_error(y2_test,y2_pred)+mean_squared_error(y3_test,y3_pred))/3)
#print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_p
# redict,y_test))

print("RMSE:",RMSE(y1_test,y2_test,y3_test,y1_pred,y2_pred,y3_pred))

from sklearn.metrics import r2_score
r2=(r2_score(y1_test,y1_pred)+r2_score(y2_test,y2_pred)+r2_score(y3_test,y3_pred))/3
print("r2:",r2)

RMSE: 43.0111289572088
r2: -54.637810952590826

=======
#실습 
# 1.다:다 앙상블을 구현하시오. =>여기서는 2:3, y3추가
'''
RMSE: 43.0111289572088
r2: -54.637810952590826
'''
# 2.predict 구하기

import numpy as np

x1=np.array([range(100),range(301,401),range(1,101)]) #100개의 데이터 3개 - 100헹 3열 input_shape(3,)
x2=np.array([range(101,201),range(411,511),range(100,200)]) #output 3

y1=np.array([range(711,811),range(1,101),range(201,301)])
y2=np.array([range(501,601),range(711,811),range(100)])
y3=np.array([range(601,701),range(811,911),range(1100,1200)])

x1=np.transpose(x1) #아래 5개 모두(100,3)
x2=np.transpose(x2)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,shuffle=False,train_size=0.8)
x2_train,x2_test,y2_train,y2_test,y3_train,y3_test=train_test_split(x2,y2,y3,shuffle=False,train_size=0.8)

#2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,concatenate,Concatenate

#모델1
input1=Input(shape=(3,)) #input layer 구성
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
merge1=concatenate([dense1,dense2]) 
#merge 합치다 첫번째 모델과 두번째 모델 중간중간 가중치 값 계산되서 마지막에 계산된 것 합치면 됨
#dense1,2줄 모두 layer, merge,middle 모두 레이어
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

#모델 분기3
output3=Dense(30)(middle1)
output3=Dense(7)(output3)
output3=Dense(7)(output3)
output3=Dense(3)(output3)


#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=[output1,output2,output3])

model.summary()
'''
#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train],[y1_train,y2_train,y3_train],
          epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가, 예측
loss=model.evaluate([x1_test,x2_test],

model.summary()
print(loss)

#답 [3462.90771484375, 2320.569580078125, 1142.33837890625, 2320.569580078125, 1142.33837890625]
#   [1번째는 대표loss,2첫번재 모델loss, 3두번재모델loss, 4첫번째 모델metrics(mse), 5두번째 모델(mse)]    
#   1번째=2번째+3번째, 1번째=4번째+5번째   #metrics가 mae일 경우, 29,38이런식으로 나옴

y1_pred,y2_pred,y3_pred=model.predict([x1_test,x2_test])
print(y1_pred,y2_pred,y3_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y1_test,y2_test,y3_test,y1_pred,y2_pred,y3_pred):
    return np.sqrt((mean_squared_error(y1_test,y1_pred)+mean_squared_error(y2_test,y2_pred)+mean_squared_error(y3_test,y3_pred))/3)
#print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_p
# redict,y_test))

print("RMSE:",RMSE(y1_test,y2_test,y3_test,y1_pred,y2_pred,y3_pred))

from sklearn.metrics import r2_score
r2=(r2_score(y1_test,y1_pred)+r2_score(y2_test,y2_pred)+r2_score(y3_test,y3_pred))/3
print("r2:",r2)

RMSE: 43.0111289572088
r2: -54.637810952590826

>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras15_ensemble3.py
'''