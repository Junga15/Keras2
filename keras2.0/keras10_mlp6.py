<<<<<<< HEAD:keras2.0/keras10_mlp6.py
#keras10_mlp6.py 실습
#1:다 mlp(일대다 다층퍼셉트론)
#과제:코드를 완성하고 mlp4처럼 predict값을 도출할 것(x_pred2값은 주어진 것)
#변경추가사항:#1.데이터 구성에서 x열이 1개로 구성

'''
x_test,y_test에 대한 로스외의 값

1차시
loss: 0.01810389757156372
mae: 0.1155942901968956

2차시
loss: 21.169872283935547
mae: 3.3767330646514893
RMSE: 4.601072881976383
R2: 0.9732137985009272

y_pred2값_1차시  ※조건:x_pred2=np.array([0,1,99])
[[6.5694263e+02 6.1323208e-01 1.8710392e+01]
 [6.5870685e+02 1.6224852e+00 1.9360758e+01]
 [8.3161115e+02 1.0052752e+02 8.3096146e+01]]

y_pred2값_2차시 =>맞을까?
[[706.5739     -2.7409987  12.132596 ]
 [707.64935    -1.6612267  12.877538 ]
 [813.03485   104.157936   85.88353  ]]

'''

import numpy as np

x=np.array(range(100))
y=np.array([range(711,811),range(1,101),range(100)])
print(x.shape) #(1,100)=(100,)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y) #(100,3)
print(x) #[0,1,...,98,99]
print(x.shape) #=>(100,1)=(100,)

x_pred2=np.array([0,1,99])
print("x_pred2.shape:",x_pred2.shape) #(3,)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=1)) #칼럼의 갯수 3
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #(100,3)이므로 나가는 칼럼은 3개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
#print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

y_pred2=model.predict(x_pred2)
print(y_pred2)

=======
#keras10_mlp6.py 실습
#1:다 mlp(일대다 다층퍼셉트론)
#과제:코드를 완성하고 mlp4처럼 predict값을 도출할 것(x_pred2값은 주어진 것)
#변경추가사항:#1.데이터 구성에서 x열이 1개로 구성

'''
x_test,y_test에 대한 로스외의 값

1차시
loss: 0.01810389757156372
mae: 0.1155942901968956

2차시
loss: 21.169872283935547
mae: 3.3767330646514893
RMSE: 4.601072881976383
R2: 0.9732137985009272

y_pred2값_1차시  ※조건:x_pred2=np.array([0,1,99])
[[6.5694263e+02 6.1323208e-01 1.8710392e+01]
 [6.5870685e+02 1.6224852e+00 1.9360758e+01]
 [8.3161115e+02 1.0052752e+02 8.3096146e+01]]

y_pred2값_2차시 =>맞을까?
[[706.5739     -2.7409987  12.132596 ]
 [707.64935    -1.6612267  12.877538 ]
 [813.03485   104.157936   85.88353  ]]

'''

import numpy as np

x=np.array(range(100))
y=np.array([range(711,811),range(1,101),range(100)])
print(x.shape) #(1,100)=(100,)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y) #(100,3)
print(x) #[0,1,...,98,99]
print(x.shape) #=>(100,1)=(100,)

x_pred2=np.array([0,1,99])
print("x_pred2.shape:",x_pred2.shape) #(3,)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=1)) #칼럼의 갯수 3
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #(100,3)이므로 나가는 칼럼은 3개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
#print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

y_pred2=model.predict(x_pred2)
print(y_pred2)

>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras10_mlp6.py
