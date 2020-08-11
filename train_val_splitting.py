import shutil
import os
import sys
import random


def split_into_train_val(dr):
    cc_set_names = set()
    bz_set_names = set()

    for root, dirs, files in os.walk(dr):
        for directory in dirs:
            if "cc" in directory:
                cc_set_names.add(directory)
            if "bz" in directory:
                bz_set_names.add(directory)

    random.shuffle(list(cc_set_names))
    random.shuffle(list(bz_set_names))

    cc = list(cc_set_names)
    bz = list(bz_set_names)

    cc_train = cc[:int(len(cc) * 0.8)]
    cc_val = cc[int(len(cc) * 0.8):]

    bz_train = bz[:int(len(bz) * 0.8)]
    bz_val = bz[int(len(bz) * 0.8):]

    return cc_train, cc_val, bz_train, bz_val


def split_by_set(dr):
    cc_set_names = set()
    bz_set_names = set()
    cc_file_names = set()
    bz_file_names = set()
    # Collect sets and files 
    for root, dirs, files in os.walk(dr):
        for f in files:
            set_name = f[4:6]
            if "cc" in f:
                cc_set_names.add(set_name)
                cc_file_names.add(f)
            if "bz" in f:
                bz_set_names.add(set_name)
                bz_file_names.add(f)

    # Split the sets 
    random.shuffle(list(cc_set_names))
    random.shuffle(list(bz_set_names))
    cc_set = list(cc_set_names)
    bz_set = list(bz_set_names)
    cc_set_train = cc_set[:int(len(cc_set) * 0.8)]
    cc_set_val = cc_set[int(len(cc_set) * 0.8):]
    bz_set_train = bz_set[:int(len(bz_set) * 0.8)]
    bz_set_val = bz_set[int(len(bz_set) * 0.8):]


    print("cc_set_train = ", cc_set_train)
    print("cc_set_val = ", cc_set_val)
    print("bz_set_train = ", bz_set_train)
    print("bz_set_val = ", bz_set_val)
    
    # Split the files
    cc_file_train = []
    cc_file_val = []
    bz_file_train = []
    bz_file_val = []
    for f in cc_file_names:
        if f[4:6] in cc_set_train: 
            cc_file_train.append(f)
        if f[4:6] in cc_set_val: 
            cc_file_val.append(f)

    for f in bz_file_names:
        if f[4:6] in bz_set_train: 
            bz_file_train.append(f)
        if f[4:6] in bz_set_val: 
            bz_file_val.append(f)            


    return cc_file_train, cc_file_val, bz_file_train, bz_file_val


# Given a direcotyr with fly images, split them to train and val,
# acfording to their set number. Teher are multiple images in eahc set.
# For example:
#
#    set-45-cc-0.p  set-51-cc-3.p  set-59-bz-3.p

    


def main():
    random.seed(0)
    cc_train, cc_val, bz_train, bz_val = split_by_set(dr='images')
    print("cc_train = " , sorted(cc_train))

    print("cc_val = ", sorted(cc_val))

    print("bz_train = " , sorted(bz_train))
    print("bz_val = ", sorted(bz_val))


    for file in cc_train:
        shutil.copyfile("images/" + file, "images/train/" + file)
    for file in cc_val:
        shutil.copyfile("images/" + file, "images/val/" + file)
    for file in bz_train:
        shutil.copyfile("images/" + file, "images/train/" + file)
    for file in bz_val:
        shutil.copyfile("images/" + file, "images/val/" + file)


    for file in cc_train:
        pkl = file.replace(".jpg", ".p")
        shutil.copyfile("boxes/" + pkl, "boxes/train/" + pkl)
    for file in cc_val:
        pkl = file.replace(".jpg", ".p")
        shutil.copyfile("boxes/" + pkl, "boxes/val/" + pkl)
    for file in bz_train:
        pkl = file.replace(".jpg", ".p")
        shutil.copyfile("boxes/" + pkl, "boxes/train/" + pkl)
    for file in bz_val:
        pkl = file.replace(".jpg", ".p")
        shutil.copyfile("boxes/" + pkl, "boxes/val/" + pkl)

                             
if __name__ == '__main__':
    main()
