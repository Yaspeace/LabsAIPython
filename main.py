import keras
import visualkeras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy

(X_train, y_train), (X_test, y_test) = mnist.load_data()
classes_num = len(np.unique(y_train))
print(y_train[0], end='=>')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(y_train[0])

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255
img_rows, img_cols = X_train.shape[1:]
X_train = X_train.reshape(len(X_train), img_rows, img_cols, 1)
X_test = X_test.reshape(len(X_test), img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

lenet = Sequential()
lenet.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape, padding='same', name='C1'))
lenet.add(AveragePooling2D(pool_size=(2, 3), strides=(1, 1), padding='valid'))
lenet.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
lenet.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
lenet.add(Conv2D(120, kernel_size=(5, 5), activation='tanh', name=''))
lenet.add(Flatten())
lenet.add(Dense(84, activation='tanh', name='FC6'))
lenet.add(Dense(10, activation='softmax', name='OUTPUT'))
lenet.compile(loss=categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
lenet.summary()
visualkeras.layered_view(lenet).show()

history = lenet.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test), verbose=1)

