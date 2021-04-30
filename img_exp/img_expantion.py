import os
import glob
import json
from PIL import Image, ImageOps

# 画像が保存されているルートディレクトリのパス
root_dir = "base_images"

categories = []

label_data = "../src/label.json"

with open(label_data, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

for label in label_data:
    categories.append(label['label'])


for idx, cat in enumerate(categories):
    base_image_dir = root_dir + "/" + cat
    image_dir = "../src/images/" + cat
    files = glob.glob(base_image_dir + "/*.jpg")

    # 行う処理
    process1 = "mirror"
    process2 = "flip"
    process3 = "contrast"
    process4 = "rotate"

    # 画像を保存するディレクトリを作る
    os.makedirs(image_dir, exist_ok=True)

    for f in files:
        img = Image.open(f)
        # 画像に任意の処理を行う
        new_img1 = ImageOps.mirror(img)
        new_img2 = ImageOps.flip(img)
        new_img3 = ImageOps.autocontrast(img, 10)
        new_img4 = img.rotate(90)

        root, ext = os.path.splitext(f)
        basename = os.path.basename(root)

        # 画像を保存する
        img.save(os.path.join(image_dir, basename + ext))
        """
        new_img1.save(os.path.join(
            image_dir, basename + "_" + process1 + ext))
        """
        new_img2.save(os.path.join(
            image_dir, basename + "_" + process2 + ext))
        new_img3.save(os.path.join(
            image_dir, basename + "_" + process3 + ext))
        new_img4.save(os.path.join(
            image_dir, basename + "_" + process4 + ext))
