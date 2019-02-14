import csv
import numpy as np

def readCSV(filename, dtype="int64", verbose=True):
    """
    Input: Str of csv filename
    Output: numpy.ndarray of Nx17
    """
    with open(filename, 'r') as foo:
        if verbose:
            print("Reading labels from {}...".format(filename))
        csvreader = csv.reader(foo, dialect='excel')
        label_mat = []
        if dtype=="int64":
            dtype = np.int64
        elif dtype=="float":
            dtype = np.float64

        for line in csvreader:
            tmp = np.asarray(line).astype(dtype).reshape((1, -1))
            label_mat.append(tmp)
        label_mat = np.concatenate(label_mat, axis=0)
    return label_mat

def writeCSV(filename, label_mat):
    """
    Input:
        filename: str
        label_mat: np.ndarray
    """
    assert isinstance(label_mat, np.ndarray), "mat format error"
    assert label_mat.ndim==2 and label_mat.shape[1]==17, "mat format error"
    with open(filename, 'w', newline="") as csvfile:
        print("Writing labels to {}...".format(filename))
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerows(label_mat)
        print("Successful!")

if __name__ == "__main__":
    y_pred_mat = readCSV("c:/users/pzq/desktop/result-merge-1202.csv")
    writeCSV("c:/users/pzq/desktop/test.csv", y_pred_mat)
    import ipdb; ipdb.set_trace()