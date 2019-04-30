from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return FSRCNN_switch(args)

class FSRCNN_switch(nn.Module):
    def __init__(self, args, conv=common.default_conv, d=56, s=12, m=4):
        super(FSRCNN_switch, self).__init__()

        # define head module
        m_head = [
            nn.Conv2d(1, d, 5, padding = 2),
            nn.PReLU(num_parameters = d),
            nn.Conv2d(d, s, 1),
            nn.PReLU(num_parameters = s)
        ]

        # define body module
        m_body = []
        for _ in range(m):
            m_body.append(nn.Conv2d(s, s, 3, padding = 1))
            m_body.append(nn.PReLU(num_parameters = s))

        # define tail module
        m_tail = [
            nn.Conv2d(s, d, 1),
            nn.PReLU(num_parameters = d),
            nn.ConvTranspose2d(d, 1, 9, stride = 3, padding = 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, mask):
        x1 = self.head(x)
        x2 = self.head(x)
        x1 = self.body(x1)
        x1 = self.tail(x1)
        x2 = self.tail(x2)

        return mask * x2 + (1-mask) * x1 

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

