import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import numpy as np

from h5dataset_onehot import H5Dataset, RoundDataset

from config import *
from utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = torch.device("cuda")

# forward function
def Forward(net, dataset, description=""):
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    y_pred_prob = torch.tensor([])
    total = len(dataLoader.dataset)
    with torch.no_grad():
        for idx, inputs in enumerate(dataLoader):
            inputs = inputs.to(device)
            # import ipdb; ipdb.set_trace()
            outputs = net(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            y_pred_prob = torch.cat([y_pred_prob, outputs.to("cpu")], dim=0)

            num_batch = inputs.shape[0]
            progress_bar(idx*dataLoader.batch_size+num_batch, total, description=description)

    return y_pred_prob

def main():
    from utils import GetAbsoluteFilePath
    model_file_list = GetAbsoluteFilePath('model')
    import tqdm
    for model_file in tqdm.tqdm(model_file_list):
        # load net
        net = torch.load(model_file)
        net.eval()
        cudnn.benchmark = True

        # dataset
        print("Using {}...".format(os.path.basename(model_file)))
        dir_path = OUT_DIR+'/'+SPECIFIC_NAME+"/prob/"
        os.mkdir(dir_path)

        if USE_TTA:
            tta_prob = []
            for i, tta_aug in enumerate(TTA_AUG):
                testset = RoundDataset(TEST_FILE, aug=tta_aug)
                y_pred_prob = Forward(net, testset, "%d %s"%(i, tta_aug))
                tta_prob.append(y_pred_prob)
            tta_prob = torch.stack(tta_prob).sum(dim=0)
            np.save(dir_path+"{}.npy".format(os.path.basename(model_file).split('.')[0]), tta_prob)
        else:
            pass

        del net

if __name__ == "__main__":
    main()