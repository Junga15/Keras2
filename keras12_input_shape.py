#validation default는 0(none) => 디폴트값이 있었다면 val_data,split안넣어도 발로스,발mae가 그냥 나와야함

'''
verbose 디폴트값이 있는지 알려면 안넣고 해보면됨, 그랬을 때 다 나왔으므로 디폴트값이 있다는 것
verbose=0 => 과정부터 모두 아무것도 안나옴
verbose=1 => 다나옴, 로스,메트릭스(mae),발로스,발매 출력 
verbose=2 => 다나옴, 로스,메트릭스(mae),발로스,발매 출력
verbose=3 => 과정은 나오나 위의 지표 아무것도 안나옴
=>verbose는 fit에서 가장 많이 나옴

'''

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
# model.add(Dense(10,input_dim=5)) #(100,5) 이므로 칼럼의 갯수인 dim=5
model.add(Dense(10,input_shape=(5,))) #
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) #(100,2)이므로 나가는 y의 피쳐,칼럼,특성은 2개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.2,verbose=0) #각 칼럼별로 20%, x중 1,2 and 11,12
                                                  #verbose가 0일때 훈련은 빨라지나 과정이 안보임
                                                  #시간이 얼마 안걸릴때 epoch가 작을 때 과정을 보면서 할 때(아래)
                                                  # 벌보스 1 

                                                  
 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test) #(5.2)
print('loss:',loss)
print('mae:',mae)


y_predict=model.predict(x_test) #y_test와 유사한 값=y_predict
print(y_predict)
#이를 통해서 RMSE와 R2를 구해서 이 모델이 잘 만들었나 못만들었나 확인
'''
loss: 0.01810389757156372
mae: 0.1155942901968956
'''
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