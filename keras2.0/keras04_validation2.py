from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array
#np.array()
#np.array()
#array()

#1.data
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=array([11,12,13,14,15])
y_test=array([11,12,13,14,15])
x_pred=array([16,17,18])

#2.model
model=Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3.compile,
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

#4.evalute,predict
result=model.evaluate(x_test,y_test,batch_size=1)
print("result:",result)

y_pred=model.predict(x_pred)
print("y_predict:",y_pred)