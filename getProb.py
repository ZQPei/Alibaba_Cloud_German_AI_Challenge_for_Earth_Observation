import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import numpy as np

from h5dataset_onehot import H5Dataset, RoundDataset

from config import *
from utils import progress_bar, GetFilePath
from timer import Timer

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = torch.device("cuda")

# forward function
def Forward(net, dataset, description=""):
    t = Timer()
    
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    y_pred_prob = torch.tensor([])
    total = len(dataLoader.dataset)
    with torch.no_grad():
        for idx, inputs in enumerate(dataLoader):
            t.tic()
            inputs = inputs.to(device)
            # import ipdb; ipdb.set_trace()
            outputs = net(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            y_pred_prob = torch.cat([y_pred_prob, outputs.to("cpu")], dim=0)

            num_batch = inputs.shape[0]
            t.toc()
            progress_bar(idx*dataLoader.batch_size+num_batch, total, description=description, time=t.total_time)

    print()
    return y_pred_prob

def main():
    out_path = OUT_DIR+'/'+SPECIFIC_NAME+"/prob/"

    folder_list = [x for x in os.listdir(MODEL_DIR)]
    folder_list.sort()
    for folder in folder_list:
        print("Predicting {}...".format(folder))
        dir_path = out_path+folder
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        prob = []
        model_file_list = GetFilePath(os.path.join(MODEL_DIR, folder))
        for model_file in model_file_list:
            # load net
            net = torch.load(model_file)
            net.eval()
            cudnn.benchmark = True

            # dataset
            print("Using model {}...".format(os.path.basename(model_file)))

            if USE_TTA:
                tta_prob = []
                for i, tta_aug in enumerate(TTA_AUG):
                    testset = RoundDataset(TESTSET_FILE, aug=tta_aug)
                    y_pred_prob = Forward(net, testset, "%d %s"%(i, tta_aug))
                    tta_prob.append(y_pred_prob)
                tta_prob = torch.stack(tta_prob).sum(dim=0)/len(TTA_AUG)
                prob.append(tta_prob)
                np.save(dir_path+"/{}.npy".format(os.path.basename(model_file).split('.')[0]), tta_prob.numpy())
            
            del net

        prob = torch.stack(prob).sum(dim=0)/len(model_file_list)
        np.save(out_path+"/{}.npy".format(folder), prob.numpy())

if __name__ == "__main__":
    main()