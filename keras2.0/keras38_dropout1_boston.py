#18.얼리스탑팅 카피
#보스턴 모델, 
'''
[기존]
loss: 20.80682373046875
mae: 2.987952470779419
RMSE: 4.561449755250767
R2: 0.7454200738228072

[드롭아웃 적용후] =>노드의 갯수를 줄이니 성능개선됨

loss: 18.089170455932617
mae: 2.7676994800567627
RMSE: 4.2531362361703335
R2: 0.7786716971782848
'''
import numpy as np 

from sklearn.datasets import load_boston

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) #(506,13) 
print(y.shape) #(506,)  

print("========================")
print(x[:5]) 
print(y[:10]) 
print(dataset.feature_names) 


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.8,random_state=66) #행을 자르는 것
#위에 순서대로, 셔플은 섞는다(True)=>랜덤때문에 자꾸바뀜, 따라서, 랜덤단수를 고정함 

from sklearn.preprocessing import MinMaxScaler #preprocessing 전처리
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

# model=Sequential()
# model.add(Dense(100,activation='relu',input_dim=13)) 
# #model.add(Dense(128,activation='relu',input_shape=(13,))) 
# model.add(Dropout(0.2)) #100개의 20%를 안쓰겠다. 80개만 쓰겠다. 각 레이어에 모두 적용가능
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(1))

a=[0.2,0.3,0.4]
b=[0.1,0.2,0.3]
c=[100,200,300]                    #=>노드의 갯수, 드롭아웃의 개수 모두 조정가능
d=['relu','linear','elu','selu','tahn']
model=Sequential()
model.add(Dense(c,activation='relu',input_dim=13)) 
#model.add(Dense(128,activation='relu',input_shape=(13,))) 
model.add(Dropout(a)) #100개의 20%를 안쓰겠다. 80개만 쓰겠다. 각 레이어에 모두 적용가능
model.add(Dense(c,activation=d))
model.add(Dropout(a)) 
model.add(Dense(c,activation=d))
model.add(Dropout(b)) 
model.add(Dense(c))


#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping #callbacks 호출하다
early_stopping=EarlyStopping(monitor='loss',patience=20,mode='min') #patience:나의 인내심은 20번(5,10,20번)을 못참는다.로스값이 최소값이 떨어지지 않는 행위를 10번 참겠다.
                                                                    #페이션스가 커질수록 에폭이 많이돔
                                                                    #mode는 min,max,auto

model.fit(x_train,y_train,epochs=2000,batch_size=8,callbacks=[early_stopping],validation_split=0.2) #얼리스탑팅,페이션스를 넣어줬기 때문에 에폭을 2천번 넣어줘도 중간에 끝남
                                                                                            

 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test,batch_size=8) #loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x_test) #y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
#print(y_predict)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE:",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2:",r2)

