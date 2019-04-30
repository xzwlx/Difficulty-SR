import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo
import pdb

dtype = torch.cuda.FloatTensor

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.switch = args.switch
        self.swift = args.swift
        self.level = args.level
        self.test_patch_size = args.test_patch_size
        self.test_branch = args.test_branch
        self.shave = args.shave
        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale, LeNet=None):
        self.idx_scale = idx_scale
        self.LeNet = LeNet
        target = self.get_model()
        if hasattr(target, 'set_scale'): target.set_scale(idx_scale)
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function=forward_function)
        elif self.chop and not self.training:
            return self.forward_chop_r(x)
        elif self.test_branch and not self.training:
            return self.model(x)
        else:
            if self.switch:
                scale = self.scale[self.idx_scale]
                classes = self.LeNet(Variable(x/255.0, requires_grad = False).type(dtype))
                class_map = F.softmax(classes, dim=1)
                values, indices = torch.max(class_map, 1)
                indices = indices.data
                if self.swift:
                    if indices[0] >= self.level:
                        return self.model(x)
                else:
                    mask = x.new(x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)
                    for n in range(mask.shape[0]):
                        if indices[n] >= self.level:
                            mask[n,:,:,:] = 1
                        else:
                            mask[n,:,:,:] = 0
                    return self.model(x, mask)
            else:
                return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs: torch.save(target.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        load_from = None
        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.get_model().url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from: self.get_model().load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        if self.input_large:
            scale = 1
        else:
            scale = self.scale[self.idx_scale]

        n_GPUs = min(self.n_GPUs, 4)
        _, _, h, w = args[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        list_x = [[
            a[:, :, 0:h_size, 0:w_size],
            a[:, :, 0:h_size, (w - w_size):w],
            a[:, :, (h - h_size):h, 0:w_size],
            a[:, :, (h - h_size):h, (w - w_size):w]
        ] for a in args]

        list_y = []
        if w_size * h_size < min_size:
            for i in range(0, 4, n_GPUs):
                x = [torch.cat(_x[i:(i + n_GPUs)], dim=0) for _x in list_x]
                y = self.model(*x)
                if not isinstance(y, list): y = [y]
                if not list_y:
                    list_y = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y):
                        _list_y.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*list_x):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not list_y:
                    list_y = [[_y] for _y in y]
                else:
                    for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        b, c, _, _ = list_y[0][0].size()
        y = [_y[0].new(b, c, h, w) for _y in list_y]
        for _list_y, _y in zip(list_y, y):
            _y[:, :, :h_half, :w_half] \
                = _list_y[0][:, :, :h_half, :w_half]
            _y[:, :, :h_half, w_half:] \
                = _list_y[1][:, :, :h_half, (w_size - w + w_half):]
            _y[:, :, h_half:, :w_half] \
                = _list_y[2][:, :, (h_size - h + h_half):, :w_half]
            _y[:, :, h_half:, w_half:] \
                = _list_y[3][:, :, (h_size - h + h_half):, (w_size - w + w_half):]

        if len(y) == 1: y = y[0]

        return y

    def forward_chop_r(self, x):
        scale = self.scale[self.idx_scale]
        step = self.test_patch_size
        shave = self.shave
        b, c, h, w = x.size()
        out = x.new(b, c, h*scale, w*scale).fill_(0)
        div = x.new(b, c, h*scale, w*scale).fill_(0)
        for h_start in range(0, h, step-shave):
            for w_start in range(0, w, step-shave):
                if h_start+step <= h and w_start+step <= w:
                    x_in = x[:,:,h_start:h_start+step,w_start:w_start+step].clone()
                    if self.switch:
                        classes = self.LeNet(Variable(x_in/255.0, requires_grad = False).type(dtype))
                        class_map = F.softmax(classes, dim=1)
                        values, indices = torch.max(class_map, 1)
                        indices = indices.data

                        mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                        for n in range(mask.shape[0]):
                            if indices[n] >= self.level:
                                mask[n,:,:,:] = 1
                            else:
                                mask[n,:,:,:] = 0
                        x_out = self.model(x_in, mask)
                    else:
                        x_out = self.model(x_in)
                    out[:,:,h_start*scale:(h_start+step)*scale,w_start*scale:(w_start+step)*scale] += x_out
                    div[:,:,h_start*scale:(h_start+step)*scale,w_start*scale:(w_start+step)*scale] += 1

        if h%step != 0:
            for w_start in range(0, w, step-shave):
                if w_start+step <= w:
                    x_in = x[:,:,h-step:h,w_start:w_start+step].clone()
                    if self.switch:
                        classes = self.LeNet(Variable(x_in/255.0, requires_grad = False).type(dtype))
                        class_map = F.softmax(classes, dim=1)
                        values, indices = torch.max(class_map, 1)
                        indices = indices.data

                        mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                        for n in range(mask.shape[0]):
                            if indices[n] >= self.level:
                                mask[n,:,:,:] = 1
                            else:
                                mask[n,:,:,:] = 0
                        x_out = self.model(x_in, mask)
                    else:
                        x_out = self.model(x_in)
                    out[:,:,(h-step)*scale:h*scale,w_start*scale:(w_start+step)*scale] += x_out
                    div[:,:,(h-step)*scale:h*scale,w_start*scale:(w_start+step)*scale] += 1

        if w%step != 0:
            for h_start in range(0, h, step-shave):
                if h_start+step <= h:
                    x_in = x[:,:,h_start:h_start+step,w-step:w].clone()
                    if self.switch:
                        classes = self.LeNet(Variable(x_in/255.0, requires_grad = False).type(dtype))
                        class_map = F.softmax(classes, dim=1)
                        values, indices = torch.max(class_map, 1)
                        indices = indices.data

                        mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                        for n in range(mask.shape[0]):
                            if indices[n] >= self.level:
                                mask[n,:,:,:] = 1
                            else:
                                mask[n,:,:,:] = 0
                        x_out = self.model(x_in, mask)
                    else:
                        x_out = self.model(x_in)
                    out[:,:,h_start*scale:(h_start+step)*scale,(w-step)*scale:w*scale] += x_out
                    div[:,:,h_start*scale:(h_start+step)*scale,(w-step)*scale:w*scale] += 1

        if h%step != 0 and w%step != 0:
            x_in = x[:,:,h-step:h,w-step:w].clone()
            if self.switch:
                classes = self.LeNet(Variable(x_in/255.0, requires_grad = False).type(dtype))
                class_map = F.softmax(classes, dim=1)
                values, indices = torch.max(class_map, 1)
                indices = indices.data

                mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                for n in range(mask.shape[0]):
                    if indices[n] >= self.level:
                        mask[n,:,:,:] = 1
                    else:
                        mask[n,:,:,:] = 0
                x_out = self.model(x_in, mask)
            else:
                x_out = self.model(x_in)
            out[:,:,(h-step)*scale:h*scale,(w-step)*scale:w*scale] += x_out
            div[:,:,(h-step)*scale:h*scale,(w-step)*scale:w*scale] += 1

        return out / div



    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

