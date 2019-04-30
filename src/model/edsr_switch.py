from model import common

import pdb
import torch.nn as nn

def make_model(args, parent=False):
    return EDSR_switch(args)

class EDSR_switch(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_switch, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        if args.n_colors == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        elif args.n_colors == 1:
            self.sub_mean = common.MeanShift(args.rgb_range, args.n_colors, rgb_mean=(0.4300,), rgb_std=(1.0,))
            self.add_mean = common.MeanShift(args.rgb_range, args.n_colors, rgb_mean=(0.4300,), rgb_std=(1.0,), sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # define switch branch
        m_branch = [
            nn.Conv2d(1, 12, 3, padding = 1),
            nn.PReLU(num_parameters = 12),
            nn.Conv2d(12, 12, 3, padding = 1),
            nn.PReLU(num_parameters = 12),
            nn.Conv2d(12, 12, 3, padding = 1),
            nn.PReLU(num_parameters = 12),
            nn.ConvTranspose2d(12, 1, 9, stride = 3, padding = 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.branch = nn.Sequential(*m_branch)

    def forward(self, x, mask):
        z = self.branch(x)
        y = self.sub_mean(x)
        y = self.head(y)

        res_y = self.body(y)
        res_y += y

        y = self.tail(res_y)
        y = self.add_mean(y)
        x = mask * z + (1 - mask) * y

        return x 

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

