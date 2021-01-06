#keras09_mlp1.py 실습
#다:1 mlp(다대일 다층퍼셉트론)

'''
loss: 2.9680666813192147e-08
mae: 0.00014995336823631078
'''

#1.데이터 구성
import numpy as np

x=np.array([[1,2,3,4,5,6,7,8,9,10],
           [11,12,13,14,15,16,17,18,19,20]])
y=np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #(2,10) 스칼라가 2개, 칼럼/열/특성이 10
print(y.shape) #(1,10)=(10,) 같은 것임 ※ keras10_mlp4.py 수업내용 y_pred2 (2,)=(1,2)

#.칼럼이 2개있는 (10,2)로 바꾸기 
x=np.transpose(x) #행렬을 바꿔주는 트랜스포스함수 사용
#print(x) #[[1,11],[2,12],[3,13],[4,14].....[9,19],[10,20]]
print(x.shape) #(10,2)
'''
y=np.transpose(y) 
print(y) #[ 1  2  3  4  5  6  7  8  9 10] =>★트랜스포스적용 함수전과 동일한 값 도출
print(y.shape) #(10,)
'''
#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense

model=Sequential()
#model.add(Dense(10,input_dim=2)) #인풋딤이 트랜스포스로 바뀐 열인 2로 적용된 것을 볼 수 있다. 
model.add(Dense(10,input_shape=(2,)))  #인풋쉐이프를 2컴마로 구성할 수도 있다. 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x,y,epochs=100,batch_size=1,validation_split=0.2) 
#validation0 split은 각 칼럼별로 20%, 위의 예시에서는 x는 1,2와 11,12, y는 1,2 (단,shuffle=False일 경우) 
# 근데 안넣어줘도 순차적으로 나옴=>물어보기!!
#shuffle의 디폴트값(명시하지 않았을때 적용되는 값)이 True이므로 순차적 추출을 원한다면 
#반드시 shuffle=False를 명시해주어야 함. 
#print(x)

 #4.평가,예측
loss,mae=model.evaluate(x,y) #loss,mae는 a,b로 바뀌어도 상관없음
print('loss:',loss)
print('mae:',mae)

y_predict=model.predict(x) #y_predict란 x값을 넣었을 때 예측되는 y값을 의미함.
print(y_predict)

'''
loss: 2.9680666813192147e-08
mae: 0.00014995336823631078
예측되는 y값
[[1.0002755]
 [2.0002167]
 [3.0001557]
 [4.000097 ]
 [5.000036 ]
 [5.999975 ]
 [6.999916 ]
 [7.999857 ]
 [8.999796 ]
 [9.999737 ]]

'''