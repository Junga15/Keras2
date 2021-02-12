<<<<<<< HEAD:keras2.0/keras10_mlp2.py
#keras10_mlp2.py 실습
# 다:1 mlp(다대일 다층퍼셉트론)
# 추가된 과제:(train_test_split을 이용하여)데이터를 train과 test로 분리하기 + RMSE,R2값 추가해서 구하기
# 변경사항:#1.데이터구성에서 range함수 이용

'''
x_test,y_test에 대한 로스외의 값
loss: 3.725290298461914e-09
mae: 4.272460864740424e-05
RMSE: 6.103515625e-05
R2: 0.9999999999952864
'''

#1.데이터 구성
import numpy as np

x=np.array([range(100)]) 
x=np.array([range(301,401)]) 
x=np.array([range(1,101)]) 

x=np.array([range(100),range(301,401),range(1,101)])
y=np.array(range(711,811))
print(x) #[[0,1,2,...97,98,99],[301,302,...,398,399,400],[1,2,3,...,98,99,100]]
print(x.shape) #(3,100)
print(y.shape) #(100,)

x=np.transpose(x)
print(x)
print(x.shape) #(100,3) 
print(y.shape) #711~810 => #(100,)

#.데이터셋을 train과 test로 분리하기
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) 
#행을 자르는 것
#train_test_split쓸 때 train_size(혹은 test_size)는 행을 분할하여 랜덤으로 train,test로 나눈다.
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

'''
print(x_test) #[[  8 309   9],[ 93 394  94],[  4 305   5],[  5 306   6]....[ 38 339  39],[ 58 359  59]]
print(y_test) #[719 804 715 .... 749 769]
'''
print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)

#.맨처음 나의 오판: 범위를 멋대로 쪼갬
# x_train=np.array([range(100),range(301,401)])
# x_test=np.array(range(1,101))
# y_train=np.array(range(711,761))
# y_test=np.array(range(766,811))

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
#model.add(Dense(10,input_dim=3)) #인풋딤이 3
model.add(Dense(10,input_shape=(3,))) #인풋쉐이프 (3,)도 가능
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #(100,)이므로 나가는 칼럼은 1개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
#print(y_predict)

# loss: 0.01810389757156372
# mae: 0.1155942901968956

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

=======
#keras10_mlp2.py 실습
# 다:1 mlp(다대일 다층퍼셉트론)
# 추가된 과제:(train_test_split을 이용하여)데이터를 train과 test로 분리하기 + RMSE,R2값 추가해서 구하기
# 변경사항:#1.데이터구성에서 range함수 이용

'''
x_test,y_test에 대한 로스외의 값
loss: 3.725290298461914e-09
mae: 4.272460864740424e-05
RMSE: 6.103515625e-05
R2: 0.9999999999952864
'''

#1.데이터 구성
import numpy as np

x=np.array([range(100)]) 
x=np.array([range(301,401)]) 
x=np.array([range(1,101)]) 

x=np.array([range(100),range(301,401),range(1,101)])
y=np.array(range(711,811))
print(x) #[[0,1,2,...97,98,99],[301,302,...,398,399,400],[1,2,3,...,98,99,100]]
print(x.shape) #(3,100)
print(y.shape) #(100,)

x=np.transpose(x)
print(x)
print(x.shape) #(100,3) 
print(y.shape) #711~810 => #(100,)

#.데이터셋을 train과 test로 분리하기
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) 
#행을 자르는 것
#train_test_split쓸 때 train_size(혹은 test_size)는 행을 분할하여 랜덤으로 train,test로 나눈다.
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜,따라서, 랜덤단수를 고정함 

'''
print(x_test) #[[  8 309   9],[ 93 394  94],[  4 305   5],[  5 306   6]....[ 38 339  39],[ 58 359  59]]
print(y_test) #[719 804 715 .... 749 769]
'''
print(x_test.shape) #(20,3)
print(y_test.shape) #(20,)

#.맨처음 나의 오판: 범위를 멋대로 쪼갬
# x_train=np.array([range(100),range(301,401)])
# x_test=np.array(range(1,101))
# y_train=np.array(range(711,761))
# y_test=np.array(range(766,811))

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
#model.add(Dense(10,input_dim=3)) #인풋딤이 3
model.add(Dense(10,input_shape=(3,))) #인풋쉐이프 (3,)도 가능
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #(100,)이므로 나가는 칼럼은 1개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test)
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test)
#print(y_predict)

# loss: 0.01810389757156372
# mae: 0.1155942901968956

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))
#print("mse:",mean_squared_error(y_test,y_predict))
#print("mse:",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras10_mlp2.py
