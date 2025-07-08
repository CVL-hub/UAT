import os
import torch
# pip install pysodmetrics
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
from time import time
from tqdm import tqdm
from datetime import datetime
import utils.metrics as Measure
from utils_lr import adjust_lr
from utils.utils import set_gpu, structure_loss, clip_gradient
from models.UAT import Network
from data import get_dataloader

import matplotlib.pyplot as plt

def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, supp_feats, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            supp_feats = supp_feats.cuda()

            s3, s2, s1, s0, loss_prob = model(images, supp_feats, y=gts, training=True)
            loss_s3 = structure_loss(s3, gts)
            loss_s2 = structure_loss(s2, gts)
            loss_s1 = structure_loss(s1, gts)
            loss_s0 = structure_loss(s0, gts)

            loss = loss_s0 + loss_s1 + loss_s2 + loss_s3 + loss_prob

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_s3:{:.4f} loss_s2:{:.4f} '
                      'loss_s1:{:.4f} loss_s0:{:.4f}  loss_prob:{:.4f}'.

                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_s3.data,
                             loss_s2.data, loss_s1.data, loss_s0.data, loss_prob.data))

        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 5 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch
        }, save_path + 'Net_Interrupt_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UAT')
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=352, help='training image size')
    parser.add_argument('--shot', type=int, default=5, help='number of referring images')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='6', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='/data/ranwanwu/Dataset/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./snapshot/', help='the path to save model params and log')
    parser.add_argument('--pvt_weights', type=str, default='./pvt_weights/pvt_v2_b4.pth', help='the path to save model params and log')
    opt = parser.parse_args()
    print(opt)

    # set the device for training
    set_gpu(opt.gpu_id)
    cudnn.benchmark = True

    start_time = time()

    model = Network(opt).cuda()

    base, body = [], []
    for name, param in model.named_parameters():
        if 'resnet' in name:
            base.append(param)   
        else:
            body.append(param)

    params_dict = [{'params': base, 'lr': opt.lr * 0.1}, {'params': body, 'lr': opt.lr}]

    optimizer = torch.optim.Adam(params_dict)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    print('load data...')
    train_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.batchsize, opt.num_workers, mode='train')
    total_step = len(train_loader)

    save_path = opt.save_root + opt.model_name + '/'
    save_logs_path = opt.save_root + 'logs/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_logs_path, exist_ok=True)

    writer = SummaryWriter(save_logs_path + opt.model_name)

    step = 0

    print("Start train...")
    for epoch in range(0, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        # schedule
        cosine_schedule.step()
        writer.add_scalar('lr_base', cosine_schedule.get_lr()[0], global_step=epoch)
        writer.add_scalar('lr_body', cosine_schedule.get_lr()[1], global_step=epoch)

        # train
        train(train_loader, model, optimizer, epoch, save_path, writer)

    end_time = time()
    print('it costs {} h to train'.format((end_time - start_time)/3600))