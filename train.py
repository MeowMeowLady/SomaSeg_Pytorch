from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import argparse
import sys
import logging
import h5dataset as data
from Visualize import make_dot
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch.backends.cudnn as cudnn
from dsn import DSN as Net
import time

DEBUG = False
VIS = False


#--------------------configurations------------------------------------------------
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a DSN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--use_gpu', dest='use_gpu', help='whether ', default=True, type=bool)
    parser.add_argument('--num_worker', dest = 'num_worker', help = 'number of workers for dataloader', default = 0, type = int)
    parser.add_argument('--iters', dest='max_iters', help='number of iterations to train', default=15000, type=int)
    parser.add_argument('--epoch', dest='max_epoch', help='number of epoches to train', default=10, type=int)
    parser.add_argument('--weights', dest='pretrained_model', help='initialize with pretrained model weights', default=None, type=str)
    parser.add_argument('--data', dest='data_path', help='dataset to train on', default='/media/dongmeng/Data/Code/SomaSeg_pytorch/data', type=str)
    parser.add_argument('--power', dest = 'power', help = 'power used for lr updating', default = 0.9, type = float)
    parser.add_argument('--lr', dest = 'base_lr', help = 'base learning rate', default = 0.001, type = float)
    parser.add_argument('--wd', dest = 'base_wd', help = 'weight decay', default = 0.0005, type = float)
    parser.add_argument('--momentum', dest = 'momentum', help = 'momentum', default = 0.9, type = float)
    parser.add_argument('--batch', dest = 'batch_size', help = 'bach size', default = 1, type = int)
    parser.add_argument('--log', dest = 'log_file', help = 'path for saving loss record', default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/log', type = str)
    #parser.add_argument('--snapshot', dest = 'snapshot', help = 'save model every snapshot iters', default = 3000, type = int)
    parser.add_argument('--output', dest = 'output', help = 'path for snapshot', default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/snapshot', type = str)
    parser.add_argument('--neg_rate', dest = 'neg_rate', help = 'the ratio of negVox to posVox', default = 10, type = int)
    parser.add_argument('--patch_size', dest = 'patch_size', help = 'the size of input for network', default = [64, 240, 240], type = list)

    if len(sys.argv) == 1:

        parser.print_help()
#        sys.exit(1)
    args = parser.parse_args()
    return args

#--------------------visualization training data and label-------------------------
def visualization(img, label):  #img and label are Variables
    img = np.squeeze(img.data.cpu().numpy())
    label = np.squeeze(label.data.cpu().numpy()) # convert Variable to tensor, to numpy
    [channel, height, width] = img.shape
    print img.shape
#    print '{} {}'.format(len(label==0), len(label==1))
    for c in range(channel):
        img_slice = img[c]
        label_slice = label[c]
        max_val = np.max(img_slice)
        min_val = np.min(img_slice)
        img_slice = np.floor(img_slice/float(max_val-min_val)*255.0)
        label_slice[label_slice==1] = 255
        label_slice[label_slice==2] = 125
        '''plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img_slice)
        plt.subplot(1, 2, 2)
        plt.imshow(label_slice)
        plt.close('all')  '''
        cv2.imshow('img', np.uint8(img_slice))
        cv2.imshow('label', np.uint8(label_slice))
        cv2.waitKey(0)


#-----------------------learning rate update policy--------------------------------
def poly_scheduler(optimizer, iteration, args):
    #update learning rate with poly policy
    power = args.power
    max_iters = args.max_iters
    for param_group in optimizer.param_groups:
        lr_mult = param_group['lr']/(args.base_lr*((1.0 - float(iteration - 1.0)/float(max_iters))**power))
        param_group['lr'] = args.base_lr*((1.0 - float(iteration)/float(max_iters))**power)*lr_mult

#------------------------validation---------------------------------------------------
def valid_net(args, valid_data_loader):
    #loss weight
    loss_weight = [5.0, 1.0, 2.0]
    #with torch.no_grad():
    loss_main = 0.0
    loss_aux1 = 0.0
    loss_aux2 = 0.0
    for iteration, (input_data, target) in enumerate(valid_data_loader):
        #prepare input data
        input_data, target = Variable(input_data), Variable(target)
        #set gpu mode
        if args.use_gpu:
            input_data = input_data.cuda()
            target = target.cuda()
        #forward
        out = net(input_data)
        [loss_seq, grad_seq] = net.crit_cross_entropy3d(out, target, loss_weight)
        loss_main += loss_seq[0].data[0]
        loss_aux1 += loss_seq[1].data[0]
        loss_aux2 += loss_seq[2].data[0]
    logging.info('{} samples completed, loss_main = {}, loss_aux1 = {}, loss_aux2 = {}'.
             format(iteration+1, loss_main/float(iteration+1), loss_aux1/float(iteration+1), loss_aux2/float(iteration+1)))
    return
#------------------------train_net------------------------------------------------
def train_net(net, train_data, valid_data, optimizer, args):
    max_iters = args.max_iters
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_worker)
    if not valid_data is None:
        valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_worker)
        

    loss_weight = [5.0, 1.0, 2.0]
    current_iter = 0
    current_epoch = 0
    loss_main_accu = 0
    loss_aux1_accu = 0
    loss_aux2_accu = 0
    #while current_iter < max_iters:
    while current_epoch < args.max_epoch:
        #set phase  
        net.train(True)      
  
        for iteration, (input_data, target) in enumerate(train_data_loader):
            if not current_iter%10:
                logging.info('Iteration {}, lr = {:.9f}'.format(current_iter, optimizer.param_groups[0]['lr']))

            #prepare input data
            input_data, target = Variable(input_data), Variable(target)

            #set gpu mode
            if args.use_gpu:
                input_data = input_data.cuda()
                target = target.cuda()

            if VIS:
                visualization(input_data, target)
            #initialize optimizer
            optimizer.zero_grad()

            #forward
            out = net(input_data)
            [loss_seq, grad_seq] = net.crit_cross_entropy3d(out, target, loss_weight)

            #compute loss and grad
            torch.autograd.backward(loss_seq, grad_seq)
            loss_main_accu += loss_seq[0].data[0]
            loss_aux1_accu += loss_seq[1].data[0]
            loss_aux2_accu += loss_seq[2].data[0]
            
            #update params
            optimizer.step()

            #display and logging  
            if not current_iter%10:
                logging.info('Iteration {}, main_loss = {:.8f}, aux1_loss = {:.8f}, aux2_loss = {:.8f}'.format(current_iter,
                             #loss_seq[0].data.cpu().numpy(), loss_seq[1].data.cpu().numpy(), loss_seq[2].data.cpu().numpy()))
                             loss_main_accu/10.0, loss_aux1_accu/10.0, loss_aux2_accu/10.0))
                loss_main_accu = 0
                loss_aux1_accu = 0
                loss_aux2_accu = 0
#                logging.info('Iteration {}, total_loss = {:.8f}'.format(current_iter, loss_total.data[0]))
    #      if current_iter + 1 == max_iters:
    #            break
            #update current_iter
            current_iter += 1

            #update lr
            poly_scheduler(optimizer, current_iter, args)
 
            #net.eval()
            #logging.info('start validation...')
            #valid_net(args, valid_data_loader)

        #update current_epoch and save model
        current_epoch += 1
        torch.save(net, os.path.join(args.output, 'model_epoch_' + '{}'.format(current_epoch) + '.pkl'))
        torch.save(net.state_dict(), os.path.join(args.output, 'epoch_' + '{}'.format(current_epoch) + '.pkl'))

        #set phase  
        net.eval()
        logging.info('start validation...')
        valid_net(args, valid_data_loader) 
    return net


#------------------------main: train the net--------------------------------------- 
if __name__ == '__main__':

    #parse the arguments 
    args = parse_args()
    #initial log file
    time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level = logging.DEBUG,
            format = '%(asctime)s [%(levelname)s] at %(filename)s, %(message)s',
            datefmt = '%Y-%m-%d(%a)%H:%M:%S', filename = os.path.join(args.log_file, 'ptlog_' + time_now + '.txt'), filemode = 'w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    args_dict = args.__dict__
    for key, value in args_dict.iteritems():
        logging.info('{} = {}'.format(key, value))

    #prepare net struvture and traing strategy    
    cudnn.benchmark = True
    net = Net()

    if not torch.cuda.is_available():
        logging.warn('there is no gpu available')

    if args.use_gpu:
        net = net.cuda()

    #g = make_dot(main, params = dict(net.named_parameters()))
    #g.view()
    para_set_list = []


    #lr_mult and wd_mult for different layers
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            para_set_list.append({'params': m.weight, 'lr': args.base_lr*1.0, 'weight_decay': args.base_wd*1.0})
            para_set_list.append({'params': m.bias, 'lr': args.base_lr*2.0, 'weight_decay': 0.0})
        if isinstance(m, nn.BatchNorm3d):
            para_set_list.append({'params': m.weight})
            para_set_list.append({'params': m.bias})
    para_set_list.append({'params': net.deconv_aux1a.weight, 'lr': args.base_lr*0.1, 'weight_decay': args.base_wd*1.0})
    para_set_list.append({'params': net.deconv_aux2a.weight, 'lr': args.base_lr*0.1, 'weight_decay': args.base_wd*1.0})
    para_set_list.append({'params': net.deconv_aux2b.weight, 'lr': args.base_lr*0.1, 'weight_decay': args.base_wd*1.0})
    para_set_list.append({'params': net.deconv1a.weight})
    para_set_list.append({'params': net.deconv1a.bias})
    para_set_list.append({'params': net.deconv2a.weight})
    para_set_list.append({'params': net.deconv2a.bias})
    para_set_list.append({'params': net.deconv3a.weight})
    para_set_list.append({'params': net.deconv3a.bias})
    optimizer = optim.SGD(para_set_list, lr = args.base_lr, momentum = args.momentum, weight_decay = args.base_wd)

    #prepare training data
    train_img_path = os.path.join(args.data_path, 'h5_data')
    train_txt_path = os.path.join(args.data_path, 'train.list')
    train_data = data.h5dataset(train_img_path, train_txt_path, args)
    valid_img_path = os.path.join(args.data_path, 'h5_data_valid')
    valid_txt_path = os.path.join(args.data_path, 'valid.list')
    valid_data = data.h5dataset(valid_img_path, valid_txt_path, args)

    #train net
    net = train_net(net, train_data, valid_data, optimizer, args)

