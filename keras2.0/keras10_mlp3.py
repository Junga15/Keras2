<<<<<<< HEAD:keras2.0/keras10_mlp3.py
#keras10_mlp3.py 실습
#다:다 mlp(다대다 다층퍼셉트론)
#변경된 사항:#1.데이터 구성에서 y의 열 2개추가
#중요사항: #2.모델구성에서 마지막 레이어(아웃풋층레이어)에서 노드가 3개★(새로 생성된 y열의 수)
#                         =>model.add(Dense(3)) #★(100,3)이므로 나가는 칼럼은 3개

'''
x_test,y_test에 대한 로스외의 값
loss: 1.5448075041391007e-09
mae: 2.9313809136510827e-05
RMSE: 3.9304039803145916e-05
R2: 0.9999999999980455
'''

#1.데이터구성
import numpy as np

x=np.array([range(100),range(301,401),range(1,101)])
y=np.array([range(711,811),range(1,101),range(100)])
print(x.shape) #(3,100)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y) #(100,3)
print(x)
print(x.shape) #=>(100,3)

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
model.add(Dense(10,input_dim=3)) #칼럼의 갯수 3
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #★(100,3)이므로 나가는 칼럼은 3개

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
=======
#keras10_mlp3.py 실습
#다:다 mlp(다대다 다층퍼셉트론)
#변경된 사항:#1.데이터 구성에서 y의 열 2개추가
#중요사항: #2.모델구성에서 마지막 레이어(아웃풋층레이어)에서 노드가 3개★(새로 생성된 y열의 수)
#                         =>model.add(Dense(3)) #★(100,3)이므로 나가는 칼럼은 3개

'''
x_test,y_test에 대한 로스외의 값
loss: 1.5448075041391007e-09
mae: 2.9313809136510827e-05
RMSE: 3.9304039803145916e-05
R2: 0.9999999999980455
'''

#1.데이터구성
import numpy as np

x=np.array([range(100),range(301,401),range(1,101)])
y=np.array([range(711,811),range(1,101),range(100)])
print(x.shape) #(3,100)
print(y.shape) #(3,100)

x=np.transpose(x)
y=np.transpose(y) #(100,3)
print(x)
print(x.shape) #=>(100,3)

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
model.add(Dense(10,input_dim=3)) #칼럼의 갯수 3
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) #★(100,3)이므로 나가는 칼럼은 3개

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
>>>>>>> e4d20bf972fb001f76ae90d52362c18532f65c9e:keras10_mlp3.py
