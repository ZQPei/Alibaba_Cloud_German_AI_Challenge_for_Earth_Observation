import torch   
import torch.nn as nn  
import torch.utils.model_zoo as model_zoo
import math

cfg = { 'vgg': [256, 256, 'M', 512, 512, 'M', 512, 512],}

class VGG(nn.Module):
    def __init__(self, vgg_name='vgg'):
        super(VGG, self).__init__()
        self.features = self._make_layer(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256,17),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
    def _make_layer(self, cfg, batch_norm=True):
        layers = []
        in_channels = 10
        for v in cfg:
            if v == 'M':
                conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
                layers += [conv2d]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



