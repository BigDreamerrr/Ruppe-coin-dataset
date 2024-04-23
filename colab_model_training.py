import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D
import keras

data_size = 554

X_train = np.empty((data_size, 300, 300, 3))
y_train = np.empty((data_size, 5))

for index in range(data_size):
  X_train[index] = np.load(fr"drive/MyDrive/Train/X/{index}.npy").astype(np.float32) / 255
  y_train[index] = np.load(fr"drive/MyDrive/Train/Y/{index}.npy")

datagen = keras.preprocessing.image.ImageDataGenerator()
iter = datagen.flow(X_train, y_train)

model = Sequential()

# convolutional layer
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                 input_shape=(300, 300, 3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(250, activation='relu'))
model.add(Dense(5, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(iter, epochs=15)

model.save(r'drive/MyDrive/model.keras')