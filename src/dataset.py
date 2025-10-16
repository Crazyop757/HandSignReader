import os
import cv2
import numpy as np
import json

train_dir = "../data/asl_alphabet_train"

classes = [d for d in os.listdir(train_dir) if os.path.isdir( os.path.join(train_dir , d))]
classes.sort()
print(classes)

X_train = []
y_train = []

IMG_SIZE = (64,64)

for label,cls in enumerate(classes):
	folder_path = os.path.join(train_dir, cls)
	for file_name in os.listdir(folder_path):		
            file_path = os.path.join(folder_path,file_name)

            img = cv2.imread(file_path)
            
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img , IMG_SIZE)
            
            img = img/255.0
            X_train.append(img)
            y_train.append(label)


label_map = {cls: label for label, cls in enumerate(classes)}
print(label_map)

X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='int32')

np.savez_compressed(
     "asl_data_mapped.npz",
     X_train=X_train, y_train=y_train
)

with open("label_map.json", "w") as f:
     json.dump(label_map, f)