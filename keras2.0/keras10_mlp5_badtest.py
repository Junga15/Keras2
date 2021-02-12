<<<<<<< HEAD:keras2.0/keras10_mlp5_badtest.py
#keras10_mlp5_badtest.py 실습
#다:다 mlp(다대다 다층퍼셉트론)
#이해 안되는 부분: #3.훈련,평가부분 train size에 대한 주석(앞부분 공부미흡,20210105)

#실습:모델 완전히 쓰레기로 만들어보기
#목표 R2: 0.5이하 / 음수는 안됨
#조건 #2.모델구성부분 layer: 5개 이상,node: 각 10개 이상 
      #3.훈련,평가 부분 batch_size: 8개 이하, epochs: 30이상

'''
x_test,y_test에 대한 로스외의 값
loss: 0.01810389757156372
mae: 0.1155942901968956
RMSE: 23.285525356035976
R2: 0.31393543675679453

y_pred2값 
[[399.54288   53.298798]]
하이퍼파라미터 튜닝: 그냥 제시된 조건대로만 적용하고, layer 한층추가함
'''
#1.데이터 구성
import numpy as np

x=np.array([range(100),range(301,401),range(1,101),range(100),range(301,401)])
y=np.array([range(711,811),range(1,101)])
print(x.shape) #(5,100)
print(y.shape) #(2,100)
x_pred2=np.array([100,402,101,100,401]) #shape (5,) 따라서 다음작업
#y_pred2는 811,101 나올 것이라 유추가능 (2,) [[811,101]]=>(1,2)로 나옴
print("x_pred2.shape:",x_pred2.shape) #transpose출력 전 #(5,)=>스칼라가 5개인 1차원=[1,2,3,4,5]

x=np.transpose(x)
y=np.transpose(y) 
#x_pred2=np.transpose(x_pred2) #출력 후 값이 (5,)이므로 아래실행
x_pred2=x_pred2.reshape(1,5)

print(x.shape) #(100,5)
print(y.shape) #(100,2)
print(x_pred2.shape)
print("x_pred2.shape:",x_pred2.shape) #transpose출력 후 #(5,) -> #(1,5)=>행무시 input_dim=5인 [[1,2,3,4,5]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,5)
print(y_test.shape) #(20,2)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=5)) #(100,5) 이므로 칼럼의 갯수인 dim=5
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2)) #(100,2)이므로 나가는 y의 피쳐,칼럼,특성은 2개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=30,batch_size=8,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12
                                                 #val_x_test(4,5),c
                                                 #train_size를 0.8로 정의했을 때 16
#print('x_validation:',x_validation): x_validation이 정의되지 않았음                                                 
# #val_x, val_y

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test) 
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) #y_test와 유사한 값=y_predict
print(y_predict)

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
=======
#keras10_mlp5_badtest.py 실습
#다:다 mlp(다대다 다층퍼셉트론)
#이해 안되는 부분: #3.훈련,평가부분 train size에 대한 주석(앞부분 공부미흡,20210105)

#실습:모델 완전히 쓰레기로 만들어보기
#목표 R2: 0.5이하 / 음수는 안됨
#조건 #2.모델구성부분 layer: 5개 이상,node: 각 10개 이상 
      #3.훈련,평가 부분 batch_size: 8개 이하, epochs: 30이상

'''
x_test,y_test에 대한 로스외의 값
loss: 0.01810389757156372
mae: 0.1155942901968956
RMSE: 23.285525356035976
R2: 0.31393543675679453

y_pred2값 
[[399.54288   53.298798]]
하이퍼파라미터 튜닝: 그냥 제시된 조건대로만 적용하고, layer 한층추가함
'''
#1.데이터 구성
import numpy as np

x=np.array([range(100),range(301,401),range(1,101),range(100),range(301,401)])
y=np.array([range(711,811),range(1,101)])
print(x.shape) #(5,100)
print(y.shape) #(2,100)
x_pred2=np.array([100,402,101,100,401]) #shape (5,) 따라서 다음작업
#y_pred2는 811,101 나올 것이라 유추가능 (2,) [[811,101]]=>(1,2)로 나옴
print("x_pred2.shape:",x_pred2.shape) #transpose출력 전 #(5,)=>스칼라가 5개인 1차원=[1,2,3,4,5]

x=np.transpose(x)
y=np.transpose(y) 
#x_pred2=np.transpose(x_pred2) #출력 후 값이 (5,)이므로 아래실행
x_pred2=x_pred2.reshape(1,5)

print(x.shape) #(100,5)
print(y.shape) #(100,2)
print(x_pred2.shape)
print("x_pred2.shape:",x_pred2.shape) #transpose출력 후 #(5,) -> #(1,5)=>행무시 input_dim=5인 [[1,2,3,4,5]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size=0.2,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

print(x_test.shape) #(20,5)
print(y_test.shape) #(20,2)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=5)) #(100,5) 이므로 칼럼의 갯수인 dim=5
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2)) #(100,2)이므로 나가는 y의 피쳐,칼럼,특성은 2개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=30,batch_size=8,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12
                                                 #val_x_test(4,5),c
                                                 #train_size를 0.8로 정의했을 때 16
#print('x_validation:',x_validation): x_validation이 정의되지 않았음                                                 
# #val_x, val_y

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test) 
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) #y_test와 유사한 값=y_predict
print(y_predict)

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
>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras10_mlp5_badtest.py
print(y_pred2)  