
[케라스]무작정 튜토리얼13- CNN(Convolution Neural Network)

model.add(Conv2D(2, (2,2), input_shape=(5,5,1)))
model.add(Conv2D(4, (2,2)))

출처: https://ebbnflow.tistory.com/138 [ebb and flow]

맨처음에 5X5크기(깊이1=흑백)로 자른 픽셀(총25장) 중
2x2(즉 4장)로 자른 것을 2장 내보낸다?


from keras.models import Sequential
from keras.layers import Flatten, Dense
 
filter_size = 32
kernel_size = (3,3)
 
from keras.layers import Conv2D, MaxPool2D
 
model = Sequential()
model.add(Conv2D(2, (2,2), input_shape=(5,5,1)))
model.add(Conv2D(4, (2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
 
model.summary(


출처: https://ebbnflow.tistory.com/138 [ebb and flow]