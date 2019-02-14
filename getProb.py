import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import numpy as np

from h5dataset_onehot import H5Dataset, Round1Dataset

from config import *
from utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = torch.device("cuda:0")

# forward function
def GetProb(net, dataLoader):
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
            progress_bar(idx*dataLoader.batch_size+num_batch, total)

    return y_pred_prob



# dataloader
testset = Round1Dataset(TEST_FILE)
testLoader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# load net
from utils import GetAbsoluteFIlePath
model_file_list = GetAbsoluteFIlePath('model')
for model_file in model_file_list:
    net = torch.load(model_file)
    net.eval()
    cudnn.benchmark = True

    y_pred_prob = GetProb(net, testLoader)
    np.save(OUT_DIR+"/{}.npy".format(os.path.basename(model_file).split('.')[0]), y_pred_prob)

    del net
