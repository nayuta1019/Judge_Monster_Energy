from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_root = "images"
img_rows = 224
img_cols = 224
categories = []

# jsonデータからラベルの情報を読み取る
label_data = os.path.join('label.json')

with open(label_data, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

for label in label_data:
    print("Label: {}, {}" .format(label['index'], label['label']))
    categories.append(label['label'])


# 全データ格納用配列
x = []
y = []

for label in label_data:
    image_dir = img_root + "/" + label['label']
    files = glob.glob(image_dir + "/*.jpg")  # ファイル名取得

    for f in files:
        img = load_img(f, target_size=(img_rows, img_cols))
        data = img_to_array(img)
        x.append(data)
        y.append(label['index'])


x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, random_state=0)

print("x: %s, y: %s" % (x.shape, y.shape))
print("Train, x: %s, y: %s" % (x_train.shape, y_train.shape))
print("Test,  x: %s, y: %s" % (x_test.shape, y_test.shape))


output_file = os.path.join('dataset.npy')
dataset = (x_train, x_test, y_train, y_test)
np.save(output_file, dataset)

"""
# 外部からデータセットを読み込む
input_data = os.path.join('dataset.npy')
x_train, x_test, y_train, y_test = np.load(input_data, allow_pickle=True)
"""
# データ型の変換,前処理
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = y_train.astype("int")
y_test = y_test.astype("int")
nb_classes = np.max(y) + 1
# one-hot表現に変換
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

print("Train, x: %s, y: %s" % (x_train.shape, y_train.shape))
print("Test,  x: %s, y: %s" % (x_test.shape, y_test.shape))


# モデルの定義
model = Sequential()

model.add(Conv2D(32, 3, input_shape=(img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))


# モデル構成の確認
model.summary()


model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=30,
                    batch_size=32, validation_data=(x_test, y_test))


# 学習状況のプロット関数の定義
def plot_history(history):

    # 分類精度の履歴をプロット
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.savefig('accuracy')
    plt.figure()
    # 損失関数の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('loss')
    plt.show()


plot_history(history)

# モデル構成のデータを保存
json_string = model.to_json()
open(os.path.join('./', 'cnn_model.json'), 'w').write(json_string)

# モデルの重みデータを保存
model.save_weights(os.path.join('./', 'cnn_model_weight.hdf5'))

# 学習後の評価
score_train = model.evaluate(x_train, y_train, verbose=0)
score_test = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', score_test[1])
print('Test loss:', score_test[0])

predictions = model.predict(x_test)  # モデルを使ってテストデータの予測をする

label_y = np.argmax(predictions, axis=1)
correct_y = np.argmax(y_test, axis=1)

print("{} -> {}" .format(label_data[correct_y[0]], label_data[label_y[0]]))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label = predictions_array[i], true_label[i]
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_data[correct_y[i]]['label'],
                                         100*np.max(predictions_array),
                                         label_data[label_y[i]]['label']),
               color=color)


def plot_value_array(i, predictions_array, true_label):

    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(nb_classes),
                       predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# X個のテスト画像, 予測されたラベル, 正解ラベルを表示.
# 正しい予測は青で, 間違った予測は赤で表示.
num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, correct_y, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, correct_y)
    _ = plt.xticks(range(len(categories)), categories, rotation=90)
plt.savefig('predict')
plt.show()
