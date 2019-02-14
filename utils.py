import os
import sys
import numpy as np

def GetFilePath(folderpath):
    file_path_list = []
    for folderpath, subfolderlist, filelist in os.walk(folderpath):
        for filename in filelist:
            filepath = os.path.join(folderpath, filename)
            file_path_list.append(filepath)
    return file_path_list

def progress_bar(current, total, description="", time=None):
    string = "\r%s\tProgress[%.1f%%]:%d/%d "%(description, 100*current/total, current, total)
    if time:
        string += "\t time:%.2fs"%(time)
    sys.stdout.write(string)
    sys.stdout.flush()

def vector_toMat(y_pred_vec):
    """
    Convert y_pred vector to matrix
    Input:
        y_pred_vec: Nx1 
    Output:
        y_pred_mat: Nx17
    """
    num_classes = y_pred_vec.max()+1
    y_pred_mat = (y_pred_vec.reshape(-1,1)==np.arange(num_classes))*1
    return y_pred_mat

if __name__ == "__main__":
    files = GetAbsoluteFilePath('model')
    print(files)