"""
只需要模型在数据集的概率计算csv，不需要模型
"""

import sys
import numpy as np
from readWriteCSV import writeCSV
from utils import vector_toMat, GetAbsoluteFilePath
from config import *

def Vote():
    base_path = OUT_DIR+'/'+SPECIFIC_NAME+'/'

    prob_file_list = GetAbsoluteFilePath(base_path+"prob")
    y_pred_prob = []
    for prob_file in prob_file_list:
        tta_prob = np.load(prob_file)
        y_pred_prob.append(tta_prob)
    y_pred_prob = np.stack(y_pred_prob).sum(axis=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_pred_mat = vector_toMat(y_pred)
    writeCSV(base_path + "result.csv", y_pred_mat)

    print(y_pred_mat.sum(axis=0))

if __name__ == "__main__":
    Vote()