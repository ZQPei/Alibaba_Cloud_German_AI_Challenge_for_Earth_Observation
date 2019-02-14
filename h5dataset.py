# -*- coding: utf-8 -*-
'''
 * @Author: ZQ.Pei 
 * @Date: 2018-11-24 23:09:48 
 * @Last Modified by:   ZQ.Pei 
 * @Last Modified time: 2018-11-24 23:09:48 
'''

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import torchvision
import random

from h5transform import H5RandomCrop, H5RandomHorizontalFlip, H5RandomVerticalFlip, H5RandomRotate

MEAN_FILE = "npy/mean_new.npy"
STD_FILE = "npy/std_new.npy"

class H5Dataset(Dataset):
    def __init__(self, h5file, istrain=False, data_mode="s2", label_mode="default"):
        super(H5Dataset, self).__init__()
        fid = h5py.File(h5file, 'r')

        self.s2 = fid['sen2']
        self.labels = fid['label']
        self.len = self.labels.shape[0]

        self.data_mode  = data_mode
        if data_mode == "s2":
            self.mean = np.load(MEAN_FILE).tolist()[-10:]
            self.std  = np.load(STD_FILE).tolist()[-10:]
        elif data_mode == "s1+s2":
            self.s1 = fid['sen1']
            idx = np.concatenate([np.array([4,5]), np.arange(8,18)]).astype(np.int)
            self.mean = np.load(MEAN_FILE)[idx].tolist()
            self.std  = np.load(STD_FILE)[idx].tolist()
        else:
            raise ValueError("data mode not found!")
        self.Normalize = torchvision.transforms.Normalize(self.mean, self.std)

        assert len(self.std) == len(self.mean), "shape mismatch error"

        self.istrain = istrain
        if istrain:
            self.randomCrop  = H5RandomCrop(32, 4)
            self.randomHflip = H5RandomHorizontalFlip()
            self.randomVflip = H5RandomVerticalFlip()
            self.randomRotate = H5RandomRotate()

        self.label_mode = label_mode
        if label_mode == "cluster":
            self._get_label = np.array([0,1,2,0,0,3,4,5,4,4, 6,7,7,8,7,7,9],dtype=np.int)

    def __getitem__(self, index):
        '''
            data: [32x32x18] numpy.ndarray

            return:
            data_torch: [18x32x32] torch.FloatTensor
        '''
        if self.data_mode == "s2":
            data = self.s2[index]
        elif self.data_mode == "s1+s2":
            s1 = self.s1[index][:,:,4:6]
            s2 = self.s2[index]
            data = np.concatenate([s1,s2],axis=-1)
            
        label = self.labels[index].argmax()
        
        data_torch = self.transform(data)

        if self.label_mode == "only_urban_rural":
            label = 0 if label < 10 else 1
        elif self.label_mode == "cluster":
            label = self.getLabel(label)

        return data_torch, label

    def __len__(self):
        return self.len

    def transform(self, data):
        """
        Input:
            data: ndarray image [32x32x18]
        Output:
            data_torch: Tensor image [18x32x32]
        """
        assert isinstance(data, np.ndarray), "data should be np.ndarray type"

        data_torch = torch.from_numpy(data).float().permute(2,0,1)
        if self.istrain:
            data_torch = self.randomCrop(data_torch)
            data_torch = self.randomHflip(data_torch)
            data_torch = self.randomVflip(data_torch)
            data_torch = self.randomRotate(data_torch)

        # maybe normalization after randomcrop or randomflip will be better
        data_torch = self.Normalize(data_torch)

        return data_torch
    
    def getLabel(self, label):
        return self._get_label[label]

class RoundDataset(Dataset):
    def __init__(self, h5file, data_mode="s2"):
        super(RoundDataset, self).__init__()
        fid = h5py.File(h5file, 'r')

        self.s1 = fid['sen1']
        self.s2 = fid['sen2']
        self.len = self.s2.shape[0]

        self.data_mode  = data_mode
        if data_mode == "s2":
            self.mean = np.load(MEAN_FILE).tolist()[-10:]
            self.std  = np.load(STD_FILE).tolist()[-10:]
        elif data_mode == "s1+s2":
            idx = np.concatenate([np.array([4,5]), np.arange(8,18)]).astype(np.int)
            self.mean = np.load(MEAN_FILE)[idx].tolist()
            self.std  = np.load(STD_FILE)[idx].tolist()
        else:
            raise ValueError("data mode not found!")
        self.Normalize = torchvision.transforms.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        '''
            data: [32x32x18] numpy.ndarray

            return:
            data_torch: [18x32x32] torch.FloatTensor
        '''
        if self.data_mode == "s2":
            data = self.s2[index]
        elif self.data_mode == "s1+s2":
            s1 = self.s1[index][:,:,4:6]
            s2 = self.s2[index]
            data = np.concatenate([s1,s2],axis=-1)
        data_torch = self.transform(data)

        return data_torch

    def __len__(self):
        return self.len

    def transform(self, data):
        """
        Input:
            data: ndarray image [32x32x18]
        Output:
            data_torch: Tensor image [18x32x32]
        """
        assert isinstance(data, np.ndarray), "data should be np.ndarray type"

        data_torch = torch.from_numpy(data).float().permute(2,0,1)
        # maybe normalization after randomcrop or randomflip will be better
        data_torch = self.Normalize(data_torch)

        return data_torch


if __name__ == "__main__":
    trainset = H5Dataset("data/training.h5", data_mode="s1+s2")
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=10)
    for idx, (input, label) in enumerate(trainLoader):
        print(type(input))
        print(input.shape)
        print(input.dtype)
        print(label)
        import ipdb; ipdb.set_trace()