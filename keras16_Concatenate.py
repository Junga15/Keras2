#concatenate

import numpy as np

x1=np.array([range(100),range(301,401),range(1,101)])
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
#output1=Dense(3)(dense1)

#모델2
input2=Input(shape=(3,)) 
dense2=Dense(10,activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
#output2=Dense(3)(dense2)

#모델병합 / concatenate: 사슬같이 잇다
from tensorflow.keras.layers import concatenate, Concatenate #소문자, 대분자
#from keras.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate 
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

#모델 선언
model=Model(inputs=[input1,input2], #2개이상은 리스트로 묶는다.[]
            outputs=[output1,output2])

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
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

print("model.metrics_names:",model.metrics_names)
#model.metrics_names: ['loss', 'dense_12_loss', 'dense_16_loss', 'dense_12_mse', 'dense_16_mse']
'''
model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           40          input_2[0][0]
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 5)            55          dense_2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           40          input_1[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 5)            30          dense_3[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            55          dense[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 5)            30          dense_4[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 10)           0           dense_1[0][0]
                                                                 dense_5[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 30)           330         concatenate[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 10)           310         dense_6[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 10)           110         dense_7[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 10)           110         dense_8[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 30)           330         dense_9[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 30)           330         dense_9[0][0]
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 7)            217         dense_13[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 7)            217         dense_10[0][0]
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 7)            56          dense_14[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 3)            24          dense_11[0][0]
__________________________________________________________________________________________________
dense_16 (Dense)                (None, 3)            24          dense_15[0][0]
==================================================================================================
'''
y1_predict,y2_predict=model.predict([x1_test,x2_test])

print("====================")
print("y1_predict: \n",y1_predict)
print("====================")
print("y2_predict: \n",y2_predict)
print("====================")
#예측답: ytest가 (20,3)이니까 20개 나옴
'''
y1_predict:
 [[845.501     83.44869  271.89594 ]
 [848.2443    83.87945  272.80054 ]
 [850.9877    84.310265 273.7051  ]
 [853.73114   84.74098  274.60974 ]
 [856.4742    85.17184  275.51422 ]
 [859.21765   85.60275  276.4188  ]
 [861.96094   86.03359  277.32343 ]
 [864.7044    86.46431  278.22806 ]
 [867.4476    86.89516  279.1327  ]
 [870.191     87.325874 280.03714 ]
 [872.9343    87.75668  280.94174 ]
 [875.67755   88.18755  281.84634 ]
 [878.42096   88.61818  282.75085 ]
 [881.1641    89.049095 283.65546 ]
 [883.9075    89.48     284.56006 ]
 [886.6508    89.91079  285.4646  ]
 [889.3941    90.3415   286.36908 ]
 [892.1375    90.77235  287.27377 ]
 [894.8807    91.20311  288.17828 ]
 [897.6242    91.633965 289.0829  ]]
====================
y2_predict:
 [[615.2317   853.62714   40.58268 ]
 [617.27716  856.3952    40.744343]
 [619.3223   859.1632    40.90613 ]
 [621.3676   861.93134   41.067936]
 [623.4127   864.69904   41.229576]
 [625.4581   867.4672    41.391403]
 [627.50323  870.23486   41.55309 ]
 [629.5487   873.0031    41.714874]
 [631.5939   875.7708    41.87654 ]
 [633.63916  878.53894   42.038284]
 [635.6843   881.30676   42.200073]
 [637.7296   884.0748    42.361843]
 [639.7749   886.84283   42.523464]
 [641.82007  889.6106    42.685318]
 [643.86536  892.3787    42.8469  ]
 [645.91064  895.14667   43.008755]
 [647.9558   897.91455   43.17039 ]
 [650.0011   900.6825    43.33221 ]
 [652.0464   903.4505    43.493908]
 [654.09174  906.2185    43.655693]]
'''

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

'''
RMSE1: 45.34097095121749
RMSE2: 41.885130911451846
RMSE: 43.61305093133467
r2_1: -60.82868110674131
r2_2: -51.76283282614914
r2: -56.295756966445225
'''