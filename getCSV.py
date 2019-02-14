"""
只需要模型在数据集的概率计算csv，不需要模型
"""

import sys
import numpy as np
from readWriteCSV import writeCSV, readCSV
from utils import vector_toMat, GetFilePath
from config import *

def Vote():
    base_path = OUT_DIR+'/'+SPECIFIC_NAME+'/'

    # 1
    prob_file_list1 = [x for x in GetFilePath(base_path+"prob") if x[-4:]=='.npy']
    y_pred_prob1 = []
    for prob_file in prob_file_list1:
        prob = np.load(prob_file)
        y_pred_prob1.append(prob)
    y_pred_prob1 = np.stack(y_pred_prob1).sum(axis=0)

    # 2
    prob_file_list2 = [x for x in GetFilePath(base_path+"score") if x[-4:]=='.csv']
    y_pred_prob2 = []
    for prob_file in prob_file_list2:
        prob = readCSV(prob_file, dtype="float", verbose=False)
        y_pred_prob2.append(prob)
    y_pred_prob2 = np.stack(y_pred_prob2).sum(axis=0)

    y_pred_prob = (y_pred_prob1+y_pred_prob2)/2
    y_pred = y_pred_prob.argmax(axis=1)
    y_pred_mat = vector_toMat(y_pred)
    writeCSV(base_path + "result.csv", y_pred_mat)

    print(y_pred_mat.sum(axis=0))

if __name__ == "__main__":
    Vote()