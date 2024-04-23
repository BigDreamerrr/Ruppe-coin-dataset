import os
import cv2
import numpy as np

dir_name = r"D:\Computer vision\Images\Rupee\Processed"
labels = { 'One' : 0, 'Two' : 1, 'Five' : 2, 'Ten' : 3, 'Twenty' : 4 }

files = os.listdir(dir_name)

X = np.zeros((len(files), 300, 300, 3))
Y = np.zeros((len(files), 5))

for index, f in enumerate(files):
    full_path = os.path.join(dir_name, f)

    label = labels[f[:f.index('_')]]
    img = cv2.imread(full_path)
    
    X[index] = img
    Y[index][label] = 1


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=104, test_size=0.3, shuffle=True, stratify=Y)

for index in range(len(X_train)):
    np.save(fr'D:\Computer vision\Images\Rupee\Train\X\{index}.npy', X_train[index])
    np.save(fr'D:\Computer vision\Images\Rupee\Train\Y\{index}.npy', y_train[index])

for index in range(len(X_test)):
    np.save(fr'D:\Computer vision\Images\Rupee\Test\X\{index}.npy', X_test[index])
    np.save(fr'D:\Computer vision\Images\Rupee\Test\Y\{index}.npy', y_test[index])

for i in range(266):
    img = np.load(fr"D:\Computer vision\Images\Rupee\Test\X\{i}.npy").astype(np.uint8)
    print(np.load(fr"D:\Computer vision\Images\Rupee\Test\Y\{i}.npy").argmax())
    cv2.imshow('img', img)
    cv2.waitKey(0)