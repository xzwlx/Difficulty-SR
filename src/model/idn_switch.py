from model import common
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
from skimage import transform

def make_model(args, parent=False):
    return IDN_switch(args)

def Inter_Bicubic(x, scale, n_colors):
    x_numpy = x.data.cpu().numpy()
    x_resize = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

    for i in range(x_numpy.shape[0]):
        if n_colors == 3:
            x_resize[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
            x_resize[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
            x_resize[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        elif n_colors == 1:
            x_resize[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)

    return  Variable(torch.from_numpy(x_resize).float().cuda(), volatile=False)

# DBlocks
class Enhancement_unit(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(Enhancement_unit, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))       
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-2*nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-2*nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        block_1 = [] 
        block_1.append(nn.Conv2d(nFeat-nFeat//self.s, nFeat, kernel_size=3, padding=1, bias=True))        
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat+nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)
        self.compress = nn.Conv2d(nFeat+nDiff, nFeat, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        feature = x_feature_shot[:,0:(self.D3-self.D3//self.s),:,:]
        feature_slice = x_feature_shot[:,(self.D3-self.D3//self.s):self.D3,:,:]
        x_feat_long = self.conv_block1(feature)
        feature_concat = torch.cat((feature_slice, x), 1)
        out = x_feat_long + feature_concat
        out = self.compress(out)
        return out


class IDN_switch(nn.Module):
    def __init__(self, args):
        super(IDN_switch, self).__init__()
        nFeat = args.nFeat
        nDiff = args.nDiff
        nFeat_slice = args.nFeat_slice
        self.scale = args.scale[0]
        self.n_colors = args.n_colors
        self.test_branch = args.test_branch
        self.swift = args.swift
        self.conv1 = nn.Conv2d(self.n_colors, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.Enhan_unit1 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit2 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit3 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit4 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        # Upsampler
        self.upsample = nn.ConvTranspose2d(nFeat, self.n_colors, stride=3, kernel_size=17, padding=8)

        # small branch for plain data
        m_branch = [
            nn.Conv2d(1, 56, 5, padding = 2),
            nn.PReLU(num_parameters = 56),
            nn.Conv2d(56, 12, 3, padding = 1),
            nn.PReLU(num_parameters = 12),
            nn.Conv2d(12, 56, 3, padding = 1),
            nn.PReLU(num_parameters = 56),
            nn.ConvTranspose2d(56, 1, 9, stride = 3, padding = 3)
        ]
        self.m_branch = nn.Sequential(*m_branch)

    def forward(self, x, mask=0):
        out_s = self.m_branch(x)
        x_bicubic = Inter_Bicubic(x, self.scale, self.n_colors)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.Enhan_unit1(x)
        x = self.Enhan_unit2(x)
        x = self.Enhan_unit3(x)
        x = self.Enhan_unit4(x)

        x_upsample = self.upsample(x, output_size=x_bicubic.size())
        out = x_upsample + x_bicubic

        if self.test_branch == 1:
            return out
        elif self.test_branch == 2 or self.swift:
            return out_s
        else:
            return mask * out_s + (1-mask) * out  

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





