import numpy as np
import csv

def readCSV(filename):
    """
    Input: Str of csv filename
    Output: numpy.ndarray of Nx17
    """
    with open(filename, 'r') as foo:
        print("Reading labels from {}...".format(filename))
        csvreader = csv.reader(foo, dialect='excel')
        label_mat = []
        for line in csvreader:
            tmp = np.asarray(line).astype(np.float64).reshape((1, -1))
            label_mat.append(tmp)
        label_mat = np.concatenate(label_mat, axis=0)
    return label_mat

def chooseProbThreshold(prob, threshold):
    assert isinstance(prob, np.ndarray), "prob type error"
    assert prob.dtype == np.float64 , "prob dtype error"
    assert prob.ndim == 2 and prob.shape[1] == 17, "prob shape error"

    matched_indices = np.where(prob.max(axis=1)>threshold)
    matched = prob[matched_indices]
    matched_labels = (matched.argmax(axis=1)[:,np.newaxis]==np.arange(17))*1
    print("Number of matched :%d/%d"%(matched.shape[0], prob.shape[0]))
    print("Number of matched of each class :", (matched>threshold).sum(axis=0))

    return matched_indices, matched_labels


testfile_a = "data/integrate_testa_softmax.csv"
testfile_b = "data/integrate_testb_softmax.csv"
testfile_c = "data/jc0203_softmax.csv"
testfile_d = "data/jc0212_temp_softmax.csv"
prob_test_a = readCSV(testfile_a)
prob_test_b = readCSV(testfile_b)
prob_test_c = readCSV(testfile_c)
prob_test_d = readCSV(testfile_d)

threshold = 0.7
matched_indices_a, matched_labels_a = chooseProbThreshold(prob_test_a, threshold)
matched_indices_b, matched_labels_b = chooseProbThreshold(prob_test_b, threshold)
matched_indices_c, matched_labels_c = chooseProbThreshold(prob_test_c, threshold)
matched_indices_d, matched_labels_d = chooseProbThreshold(prob_test_d, threshold)

if __name__ == "__main__":
    import ipdb; ipdb.set_trace()

