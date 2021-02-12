from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D 
from keras.layers import Flatten, Dense, Flatten, MaxPool2D

model=Sequential()
model.add(Conv2D(filters=10,kernel_size=(2,2),strides=1, #stride의 디폴트는 1(안쓰고도 잘돌아갔으므로) (1,2)일때 
                 padding='same',input_shape=(10,10,1))) #필터는 아웃풋임
model.add(MaxPool2D(pool_size=2)) #conv2D 다음부터 사용해야함 ()=>(10,10,10)=>(5,5,10) 반띵 #pool_size=2경우:(5,5,10)/3경우:(3,3,10)/(2,3)경우:(5,3,10)
model.add(Conv2D(9,(2,2))) #첫번째는 필터, 두번째는 커널사이즈,인풋쉐이프는 위에 것 그대로감 
# model.add(Conv2D(9,(2,3))) #4차원에서 4차원으로 간것이기 때문에 오류없음,커뮤니케이션 가능
# model.add(Conv2D(8,2))     #통상적으로 Conv2D 3개 일때 성능이 좋아짐, 컨볼루션은 특성을 추출하는 것이 때문
                           #그러나 항상 그런건 아님, 데이터들 보고 판단해야함
                           #※통상적으로 LSTM 2개 연결되면 성능이 나빠짐, 시계열 데이터가 나와서 다시 잘 들어가는 경우가 아닌 않는이상                  
#padding의 디폴트값은 'valid'=>이 크기를 유지하고 싶다 했을 때 padding='same'을 넣어주면 된다.
model.add(Flatten()) 
model.add(Dense(1))

#그림의 크기 10바이10자리 #배치사이즈: 모름 실제 데이터는 (N,10,10,1)
#커널사이즈: 우리가 자르고자 하는 사이즈
#그 그림을 2바이2로 잘랐다는 것
#v필터를 10자리 통과해서 다음 레이어층은 (9,9,10)이됨


model.summary()
#CNN의 모델은 4차원,인풋쉐이프는 3차원, 출력은 그대로 4차원
#LSTM의 모델은 3차원, 인풋쉐이프,아웃풋,출력은 2차원 따라서 중간에 덴스구성시 

#<CNN 파라미터 수 연산>


# from keras.layers import Conv2D, MaxPool2D
 
# model = Sequential()
# model.add(Conv2D(2, (2,2), input_shape=(5,5,1)))
# model.add(Conv2D(4, (2,2)))