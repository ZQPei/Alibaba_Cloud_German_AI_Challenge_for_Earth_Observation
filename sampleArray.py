# -*- coding: utf-8 -*-
'''
 * @Author: ZQ.Pei 
 * @Date: 2018-11-24 23:09:48 
 * @Last Modified by:   ZQ.Pei 
 * @Last Modified time: 2018-11-24 23:09:48 
'''

import random
import numpy as np


def sampleArray(array, k):
    """
    Randomly sample an array 
    Array can be an index array

    Input:
        array: numpy.ndarray
        k: int  (k >= 0)
    Return:
        selected array, remained array
    """
    assert isinstance(array, np.ndarray), "array should be numpy.ndarray"
    N = array.shape[0]
    assert N >= k >= 0, "k should larger than array.shape[0]"

    total = range(N)
    selected = random.sample(total, k)
    remained = [i for i in total if i not in selected]

    return array[selected], array[remained]

def sampleArray_withReplacement(array, k, t=1):
    """
    Resample an array for t times with replacement
    为了加速，先对id进行采样，最后再取出所有元素
    Input:
        array: numpy.ndarray
        k: int
        t: int
    Return:
        selected: [list]
        remained: numpy.ndarray
    """
    assert t>=1 and k>=0, "invalid value"
    assert isinstance(array, np.ndarray), "array should be numpy.ndarray"
    assert array.shape[0] >= k, "number insufficient"

    N = array.shape[0]
    selected = []
    remained_idx = set(range(N))
    total = np.arange(N)
    for i in range(t):
        sampled, _ = sampleArray(total, k)
        sampled.sort()
        remained_idx -= set(sampled)
        selected.append(array[sampled])
    remained = array[np.asarray(list(remained_idx), dtype=np.int64)]
    if len(remained_idx)==1:
        remained = np.expand_dims(remained, axis=0)
    return selected, remained

def sampleArray_withoutReplacement(array, k, t=1):
    """
    Resample an array for t times without replacement
    为了加速，先对id进行采样，最后再取出所有元素
    Input:
        array: numpy.ndarray
        k: int
        t: int
    Return:
        selected: [list]
        remained: numpy.ndarray
    """
    assert t>=1 and k>=0, "invalid value"
    assert isinstance(array, np.ndarray), "array should be numpy.ndarray"
    assert array.shape[0] >= k*t, "number insufficient"
    N = array.shape[0]
    selected = []
    remained_idx = np.arange(N)
    for i in range(t):
        sampled_idx, remained_idx = sampleArray(remained_idx, k)
        sampled_idx.sort()
        selected.append(array[sampled_idx])
    remained = array[remained_idx]
    if len(remained_idx)==1:
        remained = np.expand_dims(remained, axis=0)
    return selected, remained

def shuffleArray(array):
    """
    Shuffle a numpy array or a list along the first axis.
    Input:
        array: numpy.ndarray
    Return:
        shuffled array
    """
    np.random.shuffle(array)
    return array

def sampleArray_duplication(array, t, shuffle=False, order=False):
    """
    Sample an array with duplication to an target number t.
    Input:
        array: numpy.ndarray
        t: int (t>len(array))
    Return:
        array: duplicated array
    """
    assert not(shuffle and order), "shuffle or order cannot be True at the same time"
    assert isinstance(array, np.ndarray), "type error"
    N = array.shape[0]
    assert t>=N, "value error"
    k = t//N
    r = t%N
    duplicated = array
    if k>1:
        duplicated = array.repeat(k, axis=0)
    if r>0:
        duplicated = np.concatenate([duplicated, sampleArray(array, r)[0]])
    if shuffle:
        duplicated = shuffleArray(duplicated)
    if order:
        duplicated.sort()
    return duplicated

def sampleArray_arbitary(array, k):
    assert isinstance(array, np.ndarray), "array type error"
    
    if k > array.shape[0]:
        sampled = sampleArray_duplication(array, k)
    elif k<= array.shape[0]:
        sampled, _ = sampleArray(array, k)
    return sampled


if __name__ == "__main__":
    import time

    x = np.arange(100).reshape(50,2)
    selected, remained = sampleArray(x, 1)
    print(selected.shape)
    selected, remained = sampleArray_withReplacement(x, 0, 5)
    print(selected[0].shape, remained.shape)
    selected, remained = sampleArray_withReplacement(x, 0, 5)
    print(selected[0].shape, remained.shape)
    duplicated = sampleArray_duplication(x, 120, shuffle=True)
    print(duplicated)
    # print(np.random.shuffle.__doc__)
    import ipdb; ipdb.set_trace()

    # x = np.empty((100000, 32, 32, 10))
    # start = time.time()
    # selected, remained = sampleArray_withReplacement(x, 100, 100)
    # print("Time spent :", time.time()-start)
    # import ipdb; ipdb.set_trace()
    # start = time.time()
    # selected, remained = sampleArray_withoutReplacement(x, 100, 10)
    # print("Time spent :", time.time()-start)