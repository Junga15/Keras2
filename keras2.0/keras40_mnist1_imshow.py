#<인공지능계의 hello world라 불리는 mnist!!>

import numpy as np
import matplotlib.pyplot as plt #그래프,이미지를 보고 싶다는 것

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0]) 
print(y_train[0]) #5
print("y_train[0]:",y_train[0])
print(x_train[0].shape) #(28, 28)

#plt.imshow(x_train[0],'gray')
plt.imshow(x_train[0])
plt.show() #특성이 있을수록 밝은 색이 됨,흰색이 255 가장큼

# 0~9까지의 손글씨: 다중분류

# TensorFlow 샘플에 보면 mnist 데이터셋이 많이 등장합니다. 
# MNIST는 인공지능 연구의 권위자 LeCun교수가 만든 데이터 셋이고 
# 현재 딥러닝을 공부할 때 반드시 거쳐야할 Hello, World같은 존재입니다. 
# MNIST는 60,000개의 트레이닝 셋과 10,000개의 테스트 셋으로 이루어져 있고 
# 이중 트레이닝 셋을 학습데이터로 사용하고 테스트 셋을 신경망을 검증하는 데에 사용합니다.
# MNIST는 간단한 컴퓨터 비전 데이터 세트로, 아래와 같이 손으로 쓰여진 이미지들로 구성되어 있습니다. 
# 숫자는 0에서 1까지의 값을 갖는 고정 크기 이미지 (28x28 픽셀)로 크기 표준화되고 중심에 배치되었습니다. 
# 간단히 하기 위해 각 이미지는 평평하게되어 784 피쳐의 1-D numpy 배열로 변환되었습니다 (28 * 28).
# MNIST 데이터는 Yann LeCun의 웹사이트에서 제공합니다. 
# 편의를 위해 데이터를 자동으로 다운로드하고 설치하는 코드를 포함해 놓았습니다. 
# 코드를 다운로드 하고아래와 같이 import하거나, 그냥 안에 붙여 넣으시면 됩니다.

# MNIST 데이터베이스 는 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이며, 
# 다양한 화상 처리 시스템을 트레이닝하기 위해 일반적으로 사용된다. 
# 이 데이터베이스는 또한 기계 학습 분야의 트레이닝 및 테스트에 널리 사용된다.
# MNIST의 오리지널 데이터셋의 샘플을 재혼합하여 만들어졌다.