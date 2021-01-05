import numpy as np
from sklearn.datasets import load_iris

#1.데이터
#x,y=load_iris(return_X_y=True)

dataset=load_iris()
x=dataset.data
y=dataset.target

# 먼저 숫자값으로 변환을 위해 LabelEncoder로 변환합니다. 
encoder = LabelEncoder()
encoder.fit(y)
labels = encoder.transform(y)
# 2차원 데이터로 변환합니다. 
labels = labels.reshape(-1,1)