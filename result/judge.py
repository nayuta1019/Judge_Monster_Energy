from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import os
import json
import numpy as np

img_rows = 224
img_cols = 224

# 保存したモデルの読み込み
model = model_from_json(open('../src/cnn_model.json').read())
# 保存した重みの読み込み
model.load_weights('../src/cnn_model_weight.hdf5')

categories = []

label_data = os.path.join('../', 'src/', 'label.json')

with open(label_data, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

for label in label_data:
    print("Label: {}, {}" .format(label['index'], label['label']))
    categories.append(label['label'])

# 画像を読み込む
img_path = str(input("input image path = "))
img = image.load_img(img_path, target_size=(img_rows, img_cols, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 予測
prediction = model.predict(x)

label_y = np.argmax(prediction, axis=1)
print("predict -> ", categories[label_y[0]])
