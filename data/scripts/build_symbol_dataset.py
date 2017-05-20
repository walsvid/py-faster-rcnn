#!/usr/bin/env python
# encoding: utf-8

import os
import random

file_path = "../VOCdevkit2007/VOC2007/Annotations/"

out_file_path = "../VOCdevkit2007/VOC2007/ImageSets/Main/"

raw_files = os.listdir(file_path)

basename_of_files = []

for f in raw_files:
    basename_of_files.append(f.split(".")[0])

size_of_all_data = len(basename_of_files)

train_val_test = random.sample(basename_of_files, size_of_all_data)

split_test = int(size_of_all_data*0.1)
train_val = train_val_test[split_test:]
test = train_val_test[:split_test]

split_val = int(len(train_val)*0.11)

train = train_val[split_val:]
val = train_val[:split_val]

#split_train = int(size_of_all_data*0.5)
#split_test = int(size_of_all_data*0.1)

#train = train_val_test[:split_train]
#val = train_val_test[split_train:split_test]
#test = train_val_test[split_test:]

train_val.sort()
train.sort()
val.sort()
test.sort()

with open(os.path.join(out_file_path, "trainval.txt"), "w") as ftrainval:
    ftrainval.writelines("%s\n" % item for item in train_val)

with open(os.path.join(out_file_path, "train.txt"), "w") as ftrain:
    ftrain.writelines("%s\n" % item for item in train)

with open(os.path.join(out_file_path, "val.txt"), "w") as fval:
    fval.writelines("%s\n" % item for item in val)

with open(os.path.join(out_file_path, "test.txt"), "w") as ftest:
    ftest.writelines("%s\n" % item for item in test)

# ##########################symbol

with open(os.path.join(out_file_path, "symbol_trainval.txt"), "w") as ftrainval:
    ftrainval.writelines("%s\n" % item for item in train_val)

with open(os.path.join(out_file_path, "symbol_train.txt"), "w") as ftrain:
    ftrain.writelines("%s\n" % item for item in train)

with open(os.path.join(out_file_path, "symbol_val.txt"), "w") as fval:
    fval.writelines("%s\n" % item for item in val)

with open(os.path.join(out_file_path, "symbol_test.txt"), "w") as ftest:
    ftest.writelines("%s\n" % item for item in test)
