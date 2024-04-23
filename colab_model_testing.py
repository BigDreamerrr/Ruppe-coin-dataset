import os
import numpy as np
import keras

#266
dataset_size = 238

X_test = np.empty((dataset_size, 300, 300, 3))
y_test = np.empty((dataset_size, 5))

for index in range(dataset_size):
  X_test[index] = np.load(fr"drive/MyDrive/Test/X/{index}.npy").astype(np.float32) / 255
  y_test[index] = np.load(fr"drive/MyDrive/Test/Y/{index}.npy")

model = keras.saving.load_model(r"drive/MyDrive/model.keras")
pred = model.predict(X_test)

count = 0

print(pred.shape)
print(y_test.shape)

print(pred[0])
print(y_test[0])

print(pred[1])
print(y_test[1])

print(pred[2])
print(y_test[2])

print(pred[3])
print(y_test[3])

for index in range(dataset_size):
  pre = pred[index].argmax()
  label = y_test[index].argmax()
  if pre == label:
    count += 1

print(f"Score: {count / dataset_size}")

import keras

model = keras.saving.load_model(r"drive/MyDrive/model.keras")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)