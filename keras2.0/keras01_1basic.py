#<케라스2.0 딥러닝 기본구성>

#이 스크립트에서 알아야 할 핵심키워드: input_dim, input_shape,activation,
#compile,loss,optimizer,metrics,fit,epochs,batch_size,
#(※input_dim, input_shape에 관한 설명은 KEYWORD_Tensor.py참고)

#0.넘파이 라이브러리와 텐서플로우 불러오기
import numpy as np
import tensorflow as tf

#1.데이터불러오기,데이터구성
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. 모델구성:시퀀셜 모델
from tensorflow.keras.models import Sequential 
#함수를 불러올때는 텐서플로우,케라스에서 '모델스'로 불러온다.
from tensorflow.keas.layers import Dense 
#층을 불러올때는 텐서플로우,케라스에서 '레이어스'로 불러온다.

model=Sequential() #모델(함수)는 시퀀셜 함수를 쓰겠다. 
model.add(Dense(5,input_dim=1,activation='linear')) 
#활성화함수는 모델구성할 때 지정해줘야 함,디폴트값은 리니어
#입력층
#input_dim: 
#input_shape: 
model.add(Dense(3,activation='linear')) #은닉층1:인풋값은 인풋레이어 
model.add(Dense(4)) #은닉층2
model.add(Dense(1)) #출력층

#3.컴파일과 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
#compile:로스값,로스추적값,메트릭스값
#compile:통역,사람의 용어를 컴퓨터 용어로 바꿔주는 과정
#사용자 정의 ㄴ메트릭을 사용할 때는 compile() 함수의 metrics 인자에 함수명을 삽입합니다. 
#optimizer:로스값을 최적화하기 위한 추적
#metrics:측정하는 영역, 측정지표를 의미
#※모두를 위한 딥러닝 타이핑해놓기
model.fit(x,y,epochs=1000,batch_size=1)
#훈련은 fit으로
#epochs:
#batch_size,<배치사이즈>
#배치 사이즈가 8이면 1에포크 할 때마다 64번 훈련시킴 총 506이므로
#배치사이즈 디폴트는 32(기본적으로 32번씩 던져주라), 
#  전체 샘플 데이터 중 1개씩 불러와서 처리하고자한다면 m은 1이 됩니다. 
#  또는 전체 데이터를 임의의 m개씩 묶인 작은 그룹들로 분할하여 여러번 처리할 수도 있는데 
#  이렇게 처리하면서 기계가 학습하는 것을 미니배치 학습이라고 합니다. 
#  예를 들어 전체 데이터가 1,024개가 있을 때 m을 64로 잡는다면 전체 데이터는 16개의 그룹으로 분할됩니다. 
#  각 그룹은 총 64개의 샘플로 구성됩니다. 
#  그리고 위에서 설명한 행렬 연산을 총 16번 반복하게되고 그제서야 전체 데이터에 대한 학습이 완료됩니다. 
#  이때 64를 배치 크기(Batch size)라고 합니다.





#4.평가,예측:모델에 대한 평가,예측을 여러가지 지표로 확인

model.evaluate(x,y,batch_size=1)
#평가는 evaluate으로
#evaluate() 함수는 손실값 및 메트릭 값을 반환하는 데, 여러 메트릭을 정의 및 등록하였으므로, 여러 개의 메트릭 값을 얻을 수 있습니다.

#1)loss값 구하기
loss=model.evaluate(x,y,batch_size=1)
print("loss:",loss)            

#2)y_pred값 구하기: R2,RMSE를 구하기 위해 사용하는 경우가 많음
x_pred=np.array([4])
#result=model.predict([x])
#result=model.predict([4])
result=model.predict(x_pred)
print('result:',result) #여기서 result는 y_pred
