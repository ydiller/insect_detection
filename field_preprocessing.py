import os
from pathlib import Path
import sklearn.model_selection as model_selection
import shutil
import utils

# Create lists of the data split to train/val/test in [0.6:0.2:0.2] ratio
def train_test_split(dir_path):
    json_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            json_list.append(os.path.join(root, file))
    train, test = model_selection.train_test_split(json_list, train_size=0.8, test_size=0.2)
    train, val = model_selection.train_test_split(train, train_size=0.75, test_size=0.25)

    return train, val, test


def main():
    opt = utils.parse_flags()
    dr = opt.json_directory
    for i in range(8,9):
        train_list, val_list, test_list = train_test_split(dr)
        if os.path.exists(f"../json_annotations{i}"):
            shutil.rmtree(f"../json_annotations{i}")
        os.makedirs(f"../json_annotations{i}/train/")
        os.mkdir(f"../json_annotations{i}/val/")
        os.mkdir(f"../json_annotations{i}/test/")
        for file in train_list:
            short_file_name = os.path.basename(file)
            shutil.copyfile(file, f"../json_annotations{i}/train/{short_file_name}")
        for file in val_list:
            short_file_name = os.path.basename(file)
            shutil.copyfile(file, f"../json_annotations{i}/val/{short_file_name}")
        for file in test_list:
            short_file_name = os.path.basename(file)
            shutil.copyfile(file, f"../json_annotations{i}/test/{short_file_name}")



if __name__ == '__main__':
    main()