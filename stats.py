import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

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

count = {}

for y in Y:
    count[y.argmax()] = count.get(y.argmax(), 0) + 1

pass

