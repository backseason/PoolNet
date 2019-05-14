import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.joint_poolnet import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [8,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'PoolNet Structure')

    def test(self, test_mode=1):
        mode_name = ['edge_fuse', 'sal_fuse']
        EPSILON = 1e-8
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            if test_mode == 0:
                images = images.numpy()[0].transpose((1,2,0))
                scale = [0.5, 1, 1.5, 2] # uncomment for multi-scale testing
                # scale = [1]
                multi_fuse = np.zeros(im_size, np.float32)
                for k in range(0, len(scale)):
                    im_ = cv2.resize(images, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    im_ = im_.transpose((2, 0, 1))
                    im_ = torch.Tensor(im_[np.newaxis, ...])

                    with torch.no_grad():
                        im_ = Variable(im_)
                        if self.config.cuda:
                            im_ = im_.cuda()
                        preds = self.net(im_, mode=test_mode)
                        pred_0 = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
                        pred_1 = np.squeeze(torch.sigmoid(preds[1][1]).cpu().data.numpy())
                        pred_2 = np.squeeze(torch.sigmoid(preds[1][2]).cpu().data.numpy())
                        pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

                        pred = (pred_0 + pred_1 + pred_2 + pred_fuse) / 4
                        pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)

                        pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
                        multi_fuse += pred

                multi_fuse /= len(scale)
                multi_fuse = 255 * (1 - multi_fuse)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
            elif test_mode == 1:
                with torch.no_grad():
                    images = Variable(images)
                    if self.config.cuda:
                        images = images.cuda()
                    preds = self.net(images, mode=test_mode)
                    pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = 30000 # each batch only train 30000 iters.(This number is just a random choice...)
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) == iter_num: break
                edge_image, edge_label, sal_image, sal_label = data_batch['edge_image'], data_batch['edge_label'], data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                edge_image, edge_label, sal_image, sal_label= Variable(edge_image), Variable(edge_label), Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    edge_image, edge_label, sal_image, sal_label = edge_image.cuda(), edge_label.cuda(), sal_image.cuda(), sal_label.cuda()

                # edge part
                edge_pred = self.net(edge_image, mode=0)
                edge_loss_fuse = bce2d(edge_pred[0], edge_label, reduction='sum')
                edge_loss_part = []
                for ix in edge_pred[1]:
                    edge_loss_part.append(bce2d(ix, edge_label, reduction='sum'))
                edge_loss = (edge_loss_fuse + sum(edge_loss_part)) / (self.iter_size * self.config.batch_size)
                r_edge_loss += edge_loss.data

                # sal part
                sal_pred = self.net(sal_image, mode=1)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                loss = sal_loss + edge_loss
                r_sum_loss += loss.data

                loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_edge_loss/x_showEvery, r_sal_loss/x_showEvery, r_sum_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

