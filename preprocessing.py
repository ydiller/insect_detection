import os
from pathlib import Path
import sklearn.model_selection as model_selection
from PIL import Image


# Create lists of the data split to train/val/test in [0.8:0.2:0.2] ratio
def train_test_split(dir_path):
    cc_list = []
    bz_list = []
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            if "cc" in dir:
                cc_list.append(dir)
            if "bz" in dir:
                bz_list.append(dir)

    cc_train, cc_test = model_selection.train_test_split(
        cc_list, train_size=0.8, test_size=0.2
    )
    cc_train, cc_val = model_selection.train_test_split(
        cc_train, train_size=0.75, test_size=0.25
    )

    bz_train, bz_test = model_selection.train_test_split(
        bz_list, train_size=0.8, test_size=0.2
    )
    bz_train, bz_val = model_selection.train_test_split(
        bz_train, train_size=0.75, test_size=0.25
    )

    return cc_train, cc_val, cc_test, bz_train, bz_val, bz_test


def resize_img_files(src, dst, img_shape=(512, 1024)):
    src_p = Path(src)
    dst_p = Path(dst)
    for img_path in src_p.glob("*.jpg"):
        img = Image.open(img_path)
        img = img.resize(img_shape)
        img.save(dst_p / (img_path.stem + img_path.suffix))


def save_without_resize(src, dst):
    src_p = Path(src)
    dst_p = Path(dst)
    for img_path in src_p.glob("*.jpg"):
        img = Image.open(img_path)
        img.save(dst_p / (img_path.stem + img_path.suffix))


def main():
    cc_train, cc_val, cc_test, bz_train, bz_val, bz_test = train_test_split("/images")
    os.makedirs("/data/cc/train")
    os.mkdir("/data/cc/test")
    os.mkdir("/data/cc/val")
    os.makedirs("/data/bz/train")
    os.mkdir("/data/bz/test")
    os.mkdir("/data/bz/val")
    for dir in cc_train:
        save_without_resize("/images/" + dir, "/data/cc/train/")
    for dir in cc_val:
        save_without_resize("/images/" + dir, "/data/cc/val/")
    for dir in cc_test:
        save_without_resize("/images/" + dir, "/data/cc/test/")
    for dir in bz_train:
        save_without_resize("/images/" + dir, "/data/bz/train/")
    for dir in bz_val:
        save_without_resize("/images/" + dir, "/data/bz/val/")
    for dir in bz_test:
        save_without_resize("/images/" + dir, "/data/bz/test/")


if __name__ == "__main__":
    main()
