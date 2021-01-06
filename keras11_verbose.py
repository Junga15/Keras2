#keras11_verbose.py 실습

'''
<verbose 요약>
verbose는 #3.컴파일,훈련부분인 fit에서 수행
verbose의 디폴트값은 1
(verbose 디폴트값이 있는지 알려면 안넣고 해보면됨, 해봤을 때 다 나왔으므로 디폴트값이 1임을 알수있음)

1.verbose=0 
훈련과정이 안나옴,보여주지 않고 훈련이 빨라짐, epoch이 클때 수행
ex>1/1 [==============================] - 0s 0s/step - loss: 1.6837e-09 - mae: 3.3239e-05 나오고 바로
loss: 1.6836996241664792e-09
mae: 3.3238531614188105e-0 값나옴

2.verbose=1 (디폴트값)
모두 나옴, Epoch마다 로스loss,메트릭스지표(mae),발로스val_loss,발매val_mae가 프로그레스바[==========]에 모두 출력 
epoch이 작을때 지표 보면서 수행
ex>Epoch 3/100
64/64 [==============================] - 0s 1ms/step - loss: 1098.0681 - mae: 27.8623 - val_loss: 840.4240 - val_mae: 25.6387

3.verbose=2
지표(로스,메트릭스(mae),발로스,발매)는 모두 나오나 프로그레스바[==========]가 지워짐
ex>Epoch 3/100
64/64 - 0s - loss: 3283.5754 - mae: 41.7829 - val_loss: 2716.4902 - val_mae: 36.5954

4.verbose=3 
Epoch만 나옴, 지표 loss,mae,val_loss,val_mae,프로그레스바[==========]모두 미출력
ex>Epoch 3/100
Epoch 4/100
Epoch 5/100

※validation default는 0(none) => 디폴트값이 있었다면 val_data,split안넣어도 발로스,발mae가 그냥 나와야함
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
print("x_pred2.shape:",x_pred2.shape) #transpose출력 후 #(5,): 스칼라([칼럼,피쳐,열,특성]가 5개 -> #(1,5)=>행무시 input_dim=5인 [[1,2,3,4,5]]

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
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) #(100,2)이므로 나가는 y의 피쳐,칼럼,특성은 2개

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2,verbose=0) 
                                        
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
print(y_pred2) 
