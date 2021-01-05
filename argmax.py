import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([[1,2,3],[6,5,4],[20,10,100]])
print(np.argmax(x)) ##최대값의 순서(인덱스값) 구하기
print(np.argmax(y,axis=-1)) #-1을 넣어야 최대값의 순서(인덱스값) 구하기

