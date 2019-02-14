import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import time
import matplotlib.pyplot as plt

import numpy as np
from h5dataset import H5Dataset
from sklearn.metrics import classification_report

from net import cbam
from net import xception
from net import resnet
from net import resnext
from net import densenet
from net import se_resnext

from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "checkpoint/Xception"
use_cuda = True
start_epoch = 0
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
best_acc = 0.

# net definition
net = xception.Xception()

# compute accelerating 
cudnn.benchmark = True

net.to(device)


trainset = H5Dataset(TRAINSET_FILE, istrain=True)
# testset = H5Dataset(TESTSET_FILE, istrain=False)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=0)
# testloader = torch.utils.data.DataLoader(testset,batch_size=TEST_BATCH_SIZE, shuffle=False,num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=GAMMA, dampening=0)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2, last_epoch=-1)
print(LEARNING_RATE)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(epoch):
    print_fun("\nEpoch: %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = 50
    y_train_true = torch.tensor([]).long()
    y_train_pred = torch.tensor([]).long()
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)        
        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        schedular.step()

        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        y_train_true = torch.cat([y_train_true, labels.to("cpu")], dim=0)
        y_train_pred = torch.cat([y_train_pred, outputs.to("cpu").argmax(dim=1)], dim=0)

        if (idx+1)%interval == 0:
            end = time.time()
            print_fun("[progress:{:.1f}%]time:{:.2f}s  Loss:{:.5f}  Correct:{}/{}  Acc:{:.3f}%".format(100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct,total, 100.*correct/total))
            training_loss = 0.
            start = time.time()
        
    print_fun(classification_report(y_train_true, y_train_pred))

    record['train_loss'].append(train_loss/len(trainloader))
    record['train_err'].append(1.- correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    y_val_true = torch.tensor([]).long()
    y_val_pred = torch.tensor([]).long()
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs,labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

            y_val_true = torch.cat([y_val_true, labels.to("cpu")], dim=0)
            y_val_pred = torch.cat([y_val_pred, outputs.to("cpu").argmax(dim=1)], dim=0)

        print_fun("Testing ...")
        end = time.time()
        print_fun("[progress:{:.1f}%]time:{:.2f}s  Loss:{:.5f}  Correct:{}/{}  Acc:{:.3f}%".format(100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct,total, 100.*correct/total))  

        print_fun(classification_report(y_val_true, y_val_pred))

    record['test_loss'].append(test_loss/len(testloader))
    record['test_err'].append(1.- correct/total)

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        # print_fun("Saving parameters to checkpoint/ckpt.t7")
        print_fun("Saving pkl")
        # checkpoint = {
        #     'net_dict':net.state_dict(),
        #     'acc':acc,
        #     'epoch':epoch+1,
        #     'record':record,
        # }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # torch.save(checkpoint, './{}/ckpt.t7'.format(save_dir))
        torch.save(net,'./{}/net_{:03d}.pkl'.format(save_dir,epoch+1))

    

# plot figure
x_epoch = list(range(start_epoch))
fig = plt.figure()
fig.set_size_inches(15,5)
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch):
    global record
    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'b-', lw=1, label='train')
    ax0.plot(x_epoch, record['test_loss'], 'r-', lw=1, label='val')
    ax1.plot(x_epoch, record['train_err'], 'b-', lw=1, label='train')
    ax1.plot(x_epoch, record['test_err'], 'r-', lw=1, label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.suptitle("best_acc=%f"%best_acc)
    fig.savefig("./{}/train.jpg".format(save_dir), dpi=200)
    fig.savefig("train.jpg", dpi=200)

def print_fun(string):
    print(string)
    with open("./{}/result.txt".format(save_dir), 'a+') as foo:
        print(string, file=foo)

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print_fun("Learning rate adjusted to {}".format(lr))

def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(start_epoch, start_epoch+EPOCHES):
        torch.save(net,'./{}/xception0127.pkl'.format(save_dir))
        train(epoch)
        # test(epoch)
        # draw_curve(epoch)
        # if (epoch+1)== 28:
        #     lr_decay()
        # if (epoch+1)== 56:
        #     lr_decay() 
        # if (epoch+1)== 84:
        #     lr_decay()


if __name__ == '__main__':
    main()
