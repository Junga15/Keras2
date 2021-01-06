#keras10_mlp4.py 실습
#다:다 mlp(다대다 다층퍼셉트론)
#실습1:x는(100,5),y는(100,2) 데이터 구성하여 모델을 완성하기.
#실습2:predict의 일부값을 출력하기. =>y_pred2★값 구하기
#변경추가사항: #1.데이터 구성부분 트랜스포스함수 대신 reshape함수 적용
               #4.평가,예측부분 상세한 주석(설명): y_pred2 예측하기 + y_predict값을 통해 RMSE,R2 구하는 목적 외
#변경한 것(하이퍼 파라미터 튜닝): 에퐄을 500을 줌

'''
x_test,y_test에 대한 로스외의 값
loss: 1.9191762845593985e-08
mae: 0.00012081116437911987
RMSE: 0.0001385343333377194
R2: 0.9999999999757168

y_pred2값 
[[812.49646 101.27561]]
'''

#1.데이터 구성
import numpy as np

x=np.array([range(100),range(301,401),range(1,101),range(100),range(301,401)])
y=np.array([range(711,811),range(1,101)])
print(x.shape) #(5,100)
print(y.shape) #(2,100)

x_pred2=np.array([100,402,101,100,401]) #shape (5,) 따라서 다음작업
#y_pred2는 811,101 나올 것이라 유추가능 (2,) [[811,101]]=>(1,2)로 나옴 
print("x_pred2.shape:",x_pred2.shape) #(5,)=>스칼라가 5개인 1차원=[1,2,3,4,5] (transpose출력 전) 
                                      #(5,)=(1,5)
x=np.transpose(x)
y=np.transpose(y) 
#x_pred2=np.transpose(x_pred2) #출력 후 값이 (5,)
x_pred2=x_pred2.reshape(1,5)  #위의 트랜스포스함수와 동일
print(x.shape) #(100,5)
print(y.shape) #(100,2)
print(x_pred2.shape)
print("x_pred2.shape:",x_pred2.shape) #(5,) -> #(1,5)=>행무시 input_dim=5인 [[1,2,3,4,5]] (transpose출력 후)

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
model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.2) #각 칼럼별로 20%, x중 1,2 and 11,12
                                                  #val_x, val_y
                                                  #print(v)
 #4.평가,예측
loss,mae=model.evaluate(x_test,y_test) 
print('loss:',loss) #loss가 하나가 나오는 이유는 최종것임
print('mae:',mae)

#.y예측값을 통한 RMSE와 R2 구하기
y_predict=model.predict(x_test) #y_test와 유사한 값=y_predict
print(y_predict)
#y예측값을 구하는 목적은 y예측값을 통해 RMSE와 R2를 구해서 
#로스,mae와 함께 이 모델이 잘 만들었나 못만들었나 확인하는 지표로 사용하기 위함.(RMSE와 R2:모델평가지표)

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
#(1,2)인 [[238.99931 113.22224]] 가중치가 준비되어 있기 때문에 pred2,3,4...할 수 있다.
#[[445.96103    -3.6515467]] :잘못된 예측값,xdata가 달랐음
#[[811.2032  101.46421]]: 올바른 예측값(xdata수정후)
