#[yr]
#이 방법을 통해 시계열 데이터(RNN계열)를 받으면 x와 y를 나눌 수 있다.
#알아야할 사항,핵심키워드:len(n)함수,for()함수
#참고자료: 

import numpy as np 

a=np.array(range(1,11))
size=5

print(a) #[ 1  2  3  4  5  6  7  8  9 10]
print(len(a)) #10

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1): #행
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset]) 
    print(type(aaa))
    return np.array(aaa)

# dataset = split_x(a_size)
# print("=================")
# print(dataset)

