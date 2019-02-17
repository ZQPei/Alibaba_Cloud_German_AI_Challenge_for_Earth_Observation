import h5py
import numpy as np
from chooseTestData import readCSV


trainingfile = h5py.File("data/training.h5", 'r')
validationfile = h5py.File("data/validation.h5", 'r')
testfile_a = h5py.File("data/round1_test_a_20181109.h5")
testfile_b = h5py.File("data/round1_test_b_20190104.h5")
s1_train = trainingfile['sen1']
s1_val = validationfile['sen1']
s1_test_a = testfile_a['sen1']
s1_test_b = testfile_b['sen1']
s2_train = trainingfile['sen2']
s2_val = validationfile['sen2']
s2_test_a = testfile_a['sen2']
s2_test_b = testfile_b['sen2']
labels_train = np.asarray(trainingfile['label'])
labels_val = np.asarray(validationfile['label'])
prob_test_a = readCSV("data/integrate_testa_softmax.csv")
prob_test_b = readCSV("data/integrate_testb_softmax.csv")

testfile_c = h5py.File("data/round2_test_a_20190121.h5")
s1_test_c = testfile_c['sen1']
s2_test_c = testfile_c['sen2']
testfile_d = h5py.File("data/round2_test_b_20190211.h5")
s1_test_d = testfile_d['sen1']
s2_test_d = testfile_d['sen2']
prob_test_c = readCSV("data/jc0203_softmax.csv")
prob_test_d = readCSV("data/jc0212_temp_softmax.csv")

import sys
set_id = int(sys.argv[1])

# load npy data
idx_selected = np.load("data/idx_selected_0204_%d.npy"%set_id).item()

# extract and assemble them to training_seti.h5 File
num_classes = 17

print("Processing")
idx_from_val = []
idx_from_train = []
for i in range(num_classes):
    idx_selected_from_val = idx_selected["val"][i]
    idx_selected_from_train = idx_selected["train"][i]
    idx_from_val.append(idx_selected_from_val)
    idx_from_train.append(idx_selected_from_train)

idx_from_val = np.concatenate(idx_from_val, axis=0)
idx_from_train = np.concatenate(idx_from_train, axis=0)
idx_from_val.sort()
idx_from_train.sort()

import ipdb; ipdb.set_trace()

s1_set_new = np.concatenate([np.asarray(s1_val)[idx_from_val,:,:,:], np.asarray(s1_test_a)[:,:,:,:], np.asarray(s1_test_b)[:,:,:,:], np.asarray(s1_test_c)[:,:,:,:], np.asarray(s1_test_d)[:,:,:,:]], axis=0)
s1_set_new = np.concatenate([s1_set_new, np.asarray(s1_train)[idx_from_train,:,:,:]], axis=0)
s2_set_new = np.concatenate([np.asarray(s2_val)[idx_from_val,:,:,:], np.asarray(s2_test_a)[:,:,:,:], np.asarray(s2_test_b)[:,:,:,:], np.asarray(s2_test_c)[:,:,:,:], np.asarray(s2_test_d)[:,:,:,:]], axis=0)
s2_set_new = np.concatenate([s2_set_new, np.asarray(s2_train)[idx_from_train,:,:,:]], axis=0)
labels_set_new = np.concatenate([labels_val[idx_from_val,:].astype(np.float64), prob_test_a, prob_test_b, prob_test_c, prob_test_d, labels_train[idx_from_train,:].astype(np.float64)], axis=0)

# Saving new training dataset to H5 file
f_train_new = h5py.File("data/training_0211_test_with_prob_%d.h5"%set_id, "w")
f_train_new.create_dataset("sen1", data=s1_set_new)
f_train_new.create_dataset("sen2", data=s2_set_new)
f_train_new.create_dataset("label", data=labels_set_new)
f_train_new.close()

"""
Training set distribution:
      0       3000
      1       3000
      2       3000
      3       3000
      4       3000
      5       3000
      6       3000
      7       5000
      8       3000
      9       3000
     10       3000
     11       3000
     12       3000
     13       3000
     14       2392
     15       3000
     16       3000
"""


import ipdb; ipdb.set_trace()