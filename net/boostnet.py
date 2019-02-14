import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, l=2, is_downsample=False):
        super(BasicBlock, self).__init__()
        assert l>=2 , "error"
        self.l = l
        self.identity_flag = True if is_downsample or c_in!=c_out else False
        stride = 2 if is_downsample else 1
        for i in range(1,l+1):
            conv_name = "conv%d"%i
            c_x = c_in if i==1 else c_out
            s = stride if i==1 else 1
            self.__setattr__(conv_name , nn.Sequential(
                nn.BatchNorm2d(c_x),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_x, c_out, 3, s, padding=1, bias=False),
            ))
        if self.identity_flag:
            self.identity = nn.Sequential(
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
            )
            
    def forward(self, x):
        y = self.conv1(x)
        for i in range(2, self.l+1):
            y = self.__getattr__("conv%d"%i)(y)
        if self.identity_flag:
            x = self.identity(x)
        return x.add(y)

def _make_basic_boost(Block, c_in, c_out, N, l=2, is_downsample=False):
    block_list = []
    for i in range(N):
        if i==0:
            block = Block(c_in, c_out, l, is_downsample=is_downsample)
        else:
            block = Block(c_out, c_out, l)
        block_list += [block]
    return nn.Sequential(*block_list)

class BoostBlock(nn.Module):
    def __init__(self, c_in, c_out, C, N, l=2, is_downsample=False):
        super(BoostBlock, self).__init__()
        assert C>=1 , "error"
        self.C = C
        for i in range(1, C+1):
            boost_name = "boost%d"%i
            self.__setattr__(boost_name, _make_basic_boost(BasicBlock, c_in, c_out, N, l, is_downsample=is_downsample))
        self.identity_flag = True if is_downsample or c_in!=c_out else False
        stride = 2 if is_downsample else 1
        if self.identity_flag:
            self.identity = nn.Sequential(
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
            )

    def forward(self, x):
        y = self.boost1(x)
        for i in range(2,self.C+1):
            tmp = self.__getattr__("boost%d"%i)(x)
            y = y.add(tmp)
        if self.identity_flag:
            x = self.identity(x)
        return x.add(y)
    
class BoostNet(nn.Module):
    def __init__(self, C1,C2,C3,N1,N2,N3, k, l=2, in_channels=10, num_classes=17):
        super(BoostNet, self).__init__()
        c0,c1,c2,c3 = 16*k, 16*k, 32*k, 64*k
        self.conv1 = nn.Conv2d(in_channels, c0, 3, 1,padding=1, bias=False)
        self.conv2 = BoostBlock(c0,c1,C1,N1,l,False)
        self.conv3 = BoostBlock(c1,c2,C2,N2,l,True)
        self.conv4 = BoostBlock(c2,c3,C3,N3,l,True)
        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((8,8))
        )
        self.fc = nn.Linear(c3, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
        

if __name__ == '__main__':
    net = BoostNet(1,1,1,2,2,2,2)
    x = torch.randn(4,3,32,32)
    y = net(x)
    print(y.shape)