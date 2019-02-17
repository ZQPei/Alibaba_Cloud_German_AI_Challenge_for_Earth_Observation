import h5py
import numpy as np
from sampleArray import sampleArray, sampleArray_withReplacement, sampleArray_withoutReplacement, sampleArray_arbitary

trainingfile = h5py.File("data/training.h5", 'r')
validationfile = h5py.File("data/validation.h5", 'r')
labels_train = np.asarray(trainingfile['label'])
labels_val = np.asarray(validationfile['label'])

num_classes = 17
num_of_each_class = {
    "val": labels_val.sum(axis=0),
    "train": labels_train.sum(axis=0),
}
print("Each class num in training set :",num_of_each_class["train"])
print("Each class num in validation set :",num_of_each_class["val"])
#                       0    1      2        3      4     5       6     7       8     9     10      11      12    13      14    15    16
# in training set : [ 5068, 24431, 31693,  8651, 16493, 35290,  3269, 39326, 13584, 11954, 42902,  9514,  9165, 41377,  2392,  7898, 49359] # 352366
# in validation set : [ 256, 1254, 2353,    849,   757, 1906,    474, 3395,  1914,  860,   2287,    382,   1202, 2747,   202,   672, 2609]  # 24119
idx_of_train_each_class = []
idx_of_val_each_class = []
for i in range(num_classes):
    idx_of_train_each_class.append(np.where(labels_train[:,i]==1)[0])
    idx_of_val_each_class.append(np.where(labels_val[:,i]==1)[0])

# new training set distribution
num_from_val_each_class = np.array([
    #0    1     2     3      4     5     6    7    8     9     10    11   12    13     14   15    16
    256, 1254, 2353,  849,  757, 1906,  474, 3395, 1914,  860, 2287, 382, 1202, 2747,  202,  672, 2609
])
# 从validation中选取全部样本，从training set中对每一类进行补全，来解决样本不均衡问题
# 使得每一类都保持在3000张，第7类数量太多，选取5000张
num_from_train_each_class = np.array([
    #0    1       2     3      4    5     6    7      8     9     10   11   12    13    14   15    16
    2744, 1746,  647, 2151, 2243, 1094, 2526, 1605, 1086, 2140,  713, 2618, 1798, 253, 2190, 2328, 391 #0204
])

# import ipdb; ipdb.set_trace()

idx_selected = {"val":[], "train":[]}
for i in range(num_classes):
    print("Sampling class %d"%i)
    num_from_val = num_from_val_each_class[i]
    num_from_train = num_from_train_each_class[i]
    idx_of_val = idx_of_val_each_class[i]
    idx_of_train = idx_of_train_each_class[i]
    
    for which_set in ("val","train"):
        k = eval("num_from_"+which_set)
        idx_array = eval("idx_of_"+which_set)

        selected = sampleArray_arbitary(idx_array, k)
        idx_selected[which_set].append(selected)

import ipdb; ipdb.set_trace()
# check
labels_train = labels_train.argmax(axis=1)
for i in range(num_classes):
    if idx_selected["train"][i].shape[0] == num_from_train_each_class[i] \
                    and idx_selected["val"][i].shape[0] == num_from_val_each_class[i]:
        pass
    else:
        print("error found!!! %d"%(i))
    if i==7:
        pass
    elif (idx_selected["val"][i] == idx_selected["val"][i]).all():
        pass
    else:
        print("error found!!! %d"%(i))
    for idx, array in enumerate(idx_selected["train"][i]):
        if (labels_train[array] == i).all():
            pass
        else:
            print("error found!!! %d %d"%(i,idx))


# save
np.save("data/idx_selected_0204_1.npy", idx_selected)
# np.save("data/idx_remained.npy", idx_remained)
"""
    idx_selected: Dict with keys named val and train
        idx_selected["val"] is a list of length 17
        idx_selected["val"][i] is a list of length 8
    If there is anything wrong with the new dataset, better check these two npy file first!!!

    idx_remained: Dict with keys named val and train
        idx_remained["val"][7].shape[0] = 851
        for i in range(17): print(idx_remained["train"][i].shape[0])
            (14,)
            (10463,)
            (26517,)
            (883,)
            (5104,)
            (26538,)
            (0,)
            (26486,)
            (4896,)
            (2452,)
            (37198,)
            (682,)
            (1527,)
            (39353,)
            (0,)
            (484,)
            (46231,)

"""

# new validation set distribution



import ipdb; ipdb.set_trace()