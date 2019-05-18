from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

batch_size = 128?
epochs = 10


# input image dimensions
img_x, img_y = 33, 33

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

model = Sequential()
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape = input_shape))
model.add(Conv2D(50, kernel_size=(3,3), activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, kernel_size=(3,3), activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, kernel_size=(3,3), activation='relu')
model.add(Flatten())
model.add(Dense(2048, activation='relu', input_shape=(3456,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
# The final output only has 2 coordinates
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
