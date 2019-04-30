from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return lapsrn(args)

class lapsrn(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(lapsrn, self).__init__()

        self.scale = args.scale[0]

        # define head module
        m_head = [
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # define body module
        convt_F1 = [
            _Conv_Block() for _ in range(10)
        ]
        if self.scale == 2:
            convt_F1.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False))
            convt_F1.append(nn.LeakyReLU(0.2, inplace=True))
            self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        elif self.scale == 3:
            convt_F1.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1, bias=False))
            convt_F1.append(nn.LeakyReLU(0.2, inplace=True))
            self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=3, padding=1, bias=False)
        elif self.scale == 4:
            convt_F1.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1, bias=False))
            convt_F1.append(nn.LeakyReLU(0.2, inplace=True))
            self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
            convt_F2 = [
                _Conv_Block() for _ in range(10)
            ]
            convt_F2.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=3, padding=1, bias=False))
            convt_F2.append(nn.LeakyReLU(0.2, inplace=True))
            self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

        # define translation module
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.scale == 4:
            self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.convt_F2 = nn.Sequential(*convt_F2)

        self.head = nn.Sequential(*m_head)
        self.convt_F1 = nn.Sequential(*convt_F1)

    def forward(self, x):
        out = self.head(x)
        F1 = self.convt_F1(out)
        I1 = self.convt_I1(x)
        R1 = self.convt_R1(F1)
        HR = I1 + R1

        if self.scale == 4:
            F2 = self.convt_F2(F1)
            I2 = self.convt_I2(HR)
            R2 = self.convt_R2(F2)
            HR_4x = I2 + R2
            return HR, HR_4x
        else:
            return HR 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class _Conv_Block(nn.Module):    
    def __init__(self):
        super(_Conv_Block, self).__init__()
        
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):  
        output = self.cov_block(x)
        return output