print('Importing Libraries...')

import os
import datetime
import random
import numpy as np
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
import matplotlib.pyplot as plt

with open('x_train.npy', 'rb') as file:
    x_train = np.load(file,allow_pickle=True)
with open('y_train.npy', 'rb') as file:
    y_train = np.load(file,allow_pickle=True)

model = Sequential([
	Flatten(input_shape=[4,]),
	Dense(128,activation='relu'),
	Dropout(0.5),
	Dense(256,activation='relu'),
	Dropout(0.5),
	Dense(512,activation='relu'),
	Dropout(0.5),
	Dense(256,activation='relu'),
	Dropout(0.5),
	Dense(128,activation='relu'),
	Dropout(0.5),
	Dense(2,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

def createCallback():
	os.system('load_ext tensorboard')
	os.makedirs('logs',exist_ok=True)
	logdir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
	return TensorBoard(logdir)

calback = createCallback()
earlyStoping = EarlyStopping(monitor='loss',patience=3)

history = model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,callbacks=[calback,earlyStoping])
model.save('OpenAI-Model.h5')

plt.plot(history.history['loss'],label='loss')
plt.show()