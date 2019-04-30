import os
import math
from decimal import Decimal
from model import lenet

import utility

import torch
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import pdb
from skimage import transform as stf
from skimage import io
import numpy as np
from data import common
import cv2

dtype = torch.cuda.FloatTensor

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.level = args.level

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        
        if self.args.switch or self.args.test_branch:
            self.LeNet = lenet.make_model(args).to(torch.device('cuda'))
            # self.LeNet.load_state_dict(torch.load('/mnt/lustre/luhannan/ziwei/FSRCNN_ensemble/models/2018_11_20_23_57_8/LeNet_iter_80000'), strict=False)
            self.LeNet.load_state_dict(torch.load('/mnt/lustre/luhannan/ziwei/EDSR-PyTorch-master/experiment/lenet_new/model/model_best.pt'), strict=False)
            self.LeNet.eval()

        if self.args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.args.train_classify:
                cla_out = self.model(lr, idx_scale)
                lr_numpy = lr.data.cpu().numpy()/self.args.rgb_range
                bic = np.random.random([lr_numpy.shape[0],lr_numpy.shape[1],lr_numpy.shape[2]*self.scale[0],lr_numpy.shape[3]*self.scale[0]])

                for i in range(lr_numpy.shape[0]):
                    if lr_numpy.shape[1] == 3:
                        bic[i,0,:,:] = cv2.resize(lr_numpy[i,0,:,:], (lr_numpy.shape[3]*self.scale[0],lr_numpy.shape[2]*self.scale[0]), interpolation=cv2.INTER_CUBIC)
                        bic[i,1,:,:] = cv2.resize(lr_numpy[i,1,:,:], (lr_numpy.shape[3]*self.scale[0],lr_numpy.shape[2]*self.scale[0]), interpolation=cv2.INTER_CUBIC)
                        bic[i,2,:,:] = cv2.resize(lr_numpy[i,2,:,:], (lr_numpy.shape[3]*self.scale[0],lr_numpy.shape[2]*self.scale[0]), interpolation=cv2.INTER_CUBIC)
                    elif lr_numpy.shape[1] == 1:
                        bic[i,0,:,:] = cv2.resize(lr_numpy[i,0,:,:], (lr_numpy.shape[3]*self.scale[0],lr_numpy.shape[2]*self.scale[0]), interpolation=cv2.INTER_CUBIC)
                bic = Variable(torch.from_numpy(bic*self.args.rgb_range).float().cuda(), volatile=False)
                class_label = torch.cuda.LongTensor(lr_numpy.shape[0])
                for i in range(lr_numpy.shape[0]):
                    psnr = utility.calc_psnr(bic[i:i+1,:,:,:], hr[i:i+1,:,:,:], self.scale[0], self.args.rgb_range, all_size=1)
                    if psnr < 30:
                        class_label[i] = 0
                    elif psnr < 35:
                        class_label[i] = 1
                    elif psnr < 40:
                        class_label[i] = 2
                    elif psnr < 45:
                        class_label[i] = 3
                    else:
                        class_label[i] = 4
                loss = self.loss(cla_out, class_label)
                loss.backward()
                if self.args.gclip > 0:
                        utils.clip_grad_value_(
                            self.model.parameters(),
                            self.args.gclip
                        )
                self.optimizer.step()
            else:
                if self.args.switch:
                    sr = self.model(lr, idx_scale, self.LeNet)
                else:
                    sr = self.model(lr, idx_scale)
                if sr is not None:
                    loss = self.loss(sr, hr)
                    loss.backward()
                    if self.args.gclip > 0:
                        utils.clip_grad_value_(
                            self.model.parameters(),
                            self.args.gclip
                        )
                    self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        if self.args.test_branch or self.args.train_classify or self.args.test_classify:
            count = 0
            correct = 0
            psnr_count = np.zeros(5) 

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    if self.args.train_classify or self.args.test_classify:
                        step = self.args.test_patch_size
                        if self.args.test_random:
                            x_in, x_label = common.get_patch(
                                lr, hr, 
                                patch_size=self.args.patch_size, 
                                scale=scale, 
                                multi=False, 
                                input_large=False,
                                test_random=self.args.test_random
                            )
                            classes = self.model(x_in, idx_scale)
                            class_map = F.softmax(classes, dim=1)
                            values, indices = torch.max(class_map, 1)
                            indices = indices.data
                            x_numpy = x_in.data.cpu().numpy()/self.args.rgb_range
                            bic = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

                            for i in range(x_numpy.shape[0]):
                                if x_numpy.shape[1] == 3:
                                    bic[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                    bic[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                    bic[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                elif x_numpy.shape[1] == 1:
                                    bic[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                            bic = Variable(torch.from_numpy(bic*self.args.rgb_range).float().cuda(), volatile=False)
                            for i in range(x_numpy.shape[0]):
                                count += 1
                                psnr = utility.calc_psnr(bic[i:i+1,:,:,:], x_label[i:i+1,:,:,:], scale, self.args.rgb_range, all_size=1)
                                if self.args.test_all_class:
                                    if psnr < 30:
                                        class_label = 0
                                        psnr_count[0] += 1
                                    elif psnr < 35:
                                        class_label = 1
                                        psnr_count[1] += 1
                                    elif psnr < 40:
                                        class_label = 2
                                        psnr_count[2] += 1
                                    elif psnr < 45:
                                        class_label = 3
                                        psnr_count[3] += 1
                                    else:
                                        class_label = 4
                                        psnr_count[4] += 1
                                    if indices[i] == class_label:
                                        correct += 1
                                else:
                                    if psnr < 40 and indices[i] < 3:
                                        correct += 1
                                    elif psnr >= 40 and indices[i] >= 3:
                                        correct += 1
                                    if psnr >= 40:
                                        psnr_count[3] += 1
                        else:
                            _, _, h, w = lr.size()
                            for h_start in range(0, h, step):
                                for w_start in range(0, w, step):
                                    if h_start+step <= h and w_start+step <= w:
                                        x_in = lr[:,:,h_start:h_start+step,w_start:w_start+step].clone()
                                        x_label = hr[:,:,h_start*scale:(h_start+step)*scale,w_start*scale:(w_start+step)*scale].clone()
                                        classes = self.model(x_in, idx_scale)
                                        class_map = F.softmax(classes, dim=1)
                                        values, indices = torch.max(class_map, 1)
                                        indices = indices.data
                                        x_numpy = x_in.data.cpu().numpy()/self.args.rgb_range
                                        bic = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

                                        for i in range(x_numpy.shape[0]):
                                            if x_numpy.shape[1] == 3:
                                                bic[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                                bic[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                                bic[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                            elif x_numpy.shape[1] == 1:
                                                # bic[i,0,:,:] = stf.resize(x_numpy[i,0,:,:], (x_numpy.shape[2]*scale,x_numpy.shape[3]*scale), order=3)
                                                bic[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
                                        bic = Variable(torch.from_numpy(bic*self.args.rgb_range).float().cuda(), volatile=False)
                                        for i in range(x_numpy.shape[0]):
                                            count += 1
                                            psnr = utility.calc_psnr(bic[i:i+1,:,:,:], x_label[i:i+1,:,:,:], scale, self.args.rgb_range, all_size=1)
                                            if self.args.test_all_class:
                                                if psnr < 30:
                                                    class_label = 0
                                                    psnr_count[0] += 1
                                                elif psnr < 35:
                                                    class_label = 1
                                                    psnr_count[1] += 1
                                                elif psnr < 40:
                                                    class_label = 2
                                                    psnr_count[2] += 1
                                                elif psnr < 45:
                                                    class_label = 3
                                                    psnr_count[3] += 1
                                                else:
                                                    class_label = 4
                                                    psnr_count[4] += 1
                                                if indices[i] == class_label:
                                                    correct += 1
                                            else:
                                                if psnr < 40 and indices[i] < 3:
                                                    correct += 1
                                                elif psnr >= 40 and indices[i] >= 3:
                                                    correct += 1
                                                if psnr >= 40:
                                                    psnr_count[3] += 1
                                                    # io.imsave(os.path.join('/mnt/lustre/luhannan/ziwei/test_psnr', str(int(psnr_count[3]))+'.png'), x_numpy[i,0,:,:])
                                            
                    elif self.args.test_branch:
                        step = self.args.test_patch_size
                        _, _, h, w = lr.size()
                        for h_start in range(0, h, step):
                            for w_start in range(0, w, step):
                                if h_start+step <= h and w_start+step <= w:
                                    x_in = lr[:,:,h_start:h_start+step,w_start:w_start+step].clone()
                                    x_label = hr[:,:,h_start*scale:(h_start+step)*scale,w_start*scale:(w_start+step)*scale].clone()
                                    classes = self.LeNet(x_in)
                                    class_map = F.softmax(classes, dim=1)
                                    values, indices = torch.max(class_map, 1)
                                    indices = indices.data
                                    if self.args.test_branch == 1 and indices[0] < self.level:
                                        mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                                        mask[:,:,:,:] = 0
                                        x_out = self.model(x_in, idx_scale, mask=mask)
                                        count += 1
                                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                            x_out, x_label, scale, self.args.rgb_range, dataset=d
                                        )
                                    elif self.args.test_branch == 2 and indices[0] >= self.level:
                                        mask = x_in.new(x_in.shape[0], x_in.shape[1], x_in.shape[2]*scale, x_in.shape[3]*scale)
                                        mask[:,:,:,:] = 1
                                        x_out = self.model(x_in, idx_scale, mask=mask)
                                        count += 1
                                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                            x_out, x_label, scale, self.args.rgb_range, dataset=d
                                        )
                    else:
                        if self.args.switch:
                            sr = self.model(lr, idx_scale, self.LeNet)
                        else:
                            sr = self.model(lr, idx_scale)
                        # _, _, h, w = sr.size()
                        # shave = 6
                        # sr = sr[:,:,shave:h-shave,shave:w-shave].clone()
                        # hr = hr[:,:,shave:h-shave,shave:w-shave].clone()
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                if self.args.train_classify or self.args.test_classify:
                    self.ckp.log[-1, idx_data, idx_scale] = correct / count
                elif self.args.test_branch:
                    self.ckp.log[-1, idx_data, idx_scale] /= count
                else:
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                if self.args.train_classify or self.args.test_classify:
                    self.ckp.write_log(
                        '[{} x{}]\tacc: {:.3f} \trate:{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            psnr_count[0] / count,
                            psnr_count[1] / count,
                            psnr_count[2] / count,
                            psnr_count[3] / count,
                            psnr_count[4] / count,
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results: self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

