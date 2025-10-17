import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split

train_dir = "../data/asl_alphabet_train"


classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
classes.sort()
print(f"Found {len(classes)} classes:", classes)

IMG_SIZE = (64, 64)
X = []
y = []


for label, cls in enumerate(classes):
    folder_path = os.path.join(train_dir, cls)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        img = cv2.imread(file_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0

        X.append(img)
        y.append(label)


X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

print(f"Total images loaded: {len(X)}")


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


np.savez_compressed(
    "asl_data_mapped.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)

label_map = {cls: label for label, cls in enumerate(classes)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("Dataset saved successfully!")
print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
