from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import argparse
import sys
import logging
import MyDataset as data
from Visualize import make_dot
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

DEBUG = True

#----------------------create the net stucture-----------------------------------
class DSN_net(nn.Module): 
    def __init__(self):
        super(DSN_net, self).__init__()
        self.conv1a = nn.Conv3d(1, 32, 3, 1, 1, bias = True)
        self.bn1a = nn.BatchNorm3d(32, affine = False)
        self.pool1 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv2a = nn.Conv3d(32, 64, 3, 1, 1, bias = True)
        self.bn2a = nn.BatchNorm3d(64, affine = False)
        self.conv2b = nn.Conv3d(64, 64, 3, 1, 1, bias = True)
        self.bn2b = nn.BatchNorm3d(64, affine = False)
        self.pool2 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv3a = nn.Conv3d(64, 128, 3, 1, 1, bias = True)
        self.bn3a = nn.BatchNorm3d(128, affine = False)
        self.conv3b = nn.Conv3d(128, 128, 3, 1, 1, bias = True)
        self.bn3b = nn.BatchNorm3d(128, affine = False)
        self.pool3 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv4a = nn.Conv3d(128, 256, 3, 1, 1, bias = True)
        self.bn4a = nn.BatchNorm3d(256, affine = False)
        self.conv3b = nn.Conv3d(256, 256, 3, 1, 1, bias = True)
        self.bn4b = nn.BatchNorm3d(256, affine = False)
        self.deconv1a = nn.ConvTranspose3d(256, 128, 4, 2, 1, bias = True)
        self.deconv2a = nn.ConvTranspose3d(128, 64, 4, 2, 1, bias = True)
        self.deconv3a = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = True)
        self.score_main = nn.Conv3d(32, 2, 1, 1, bias = True)
        self.deconv_aux1a = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = False)
        self.score_aux1 = nn.Conv3d(32, 2, 1, bias = True)
        self.deconv_aux2a = nn.ConvTranspose3d(128, 64, 4, 2, 1, bias = False)
        self.deconv_aux2b = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = False)
        self.score_aux2 = nn.Conv3d(32, 2, 1, bias = True)
    def forward(self, main):
        main = self.pool1(nnf.relu(self.bn1a(self.conv1a(main))))
        main = nnf.relu(self.bn2a(self.conv2a(main)))
        main = nnf.relu(self.bn2b(self.conv2b(main)))
        aux1 = self.score_aux1(self.deconv_aux1a(main))
        #make channels the first axis 
        aux1 = aux1.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        aux1 = aux1.view(aux1.numel()//2, 2)
        aux1 = nnf.log_softmax(aux1, dim=1)
        return aux1
'''        main = self.pool2(main)
        main = nnf.relu(self.bn3a(self.conv3a(main)))
        main = nnf.relu(self.bn3b(self.conv3b(main)))
        aux2 = self.score_aux2(self.deconv_aux2b(self.deconv_aux2a(main)))
        #make channels the first axis 
        aux2 = aux2.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        aux2 = aux2.view(aux2.numel()//2, 2)
        aux2 = nnf.log_softmax(aux2, dim=1)
        main = self.pool3(main)
        main = nnf.relu(self.bn4a(self.conv4a(main)))
        main = nnf.relu(self.bn4b(self.conv4b(main)))
        main = self.deconv3a(self.deconv2a(self.deconv1a(main)))
        main = self.score_main(main)
        #make channels the first axis 
        main = main.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        main = main.view(main.numel()//2, 2)
        main = nnf.log_softmax(main, dim=1)
        return main, aux1, aux2
'''
#--------------------visualization training data and label-------------------------
def visualization(img, label):  #img and label are Variables
    img = np.squeeze(img.data.cpu().numpy())
    label = np.squeeze(label.data.cpu().numpy()) # convert Variable to tensor, to numpy
    [height, width, channel] = img.shape
    #print img.size()
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



#--------------------configurations------------------------------------------------
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a DSN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--use_gpu', dest='use_gpu',
                        help='whether ',
                        default=True, type=bool)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=15000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--data', dest='data_path',
                        help='dataset to train on',
                        default='/media/dongmeng/Data/Code/SomaSeg/data/160/', type=str)
    parser.add_argument('--power', dest = 'power',
                        help = 'power used for lr updating',
                        default = 0.9, type = float)
    parser.add_argument('--lr', dest = 'base_lr',
                        help = 'base learning rate',
                        default = 0.001, type = float)
    parser.add_argument('--wd', dest = 'base_wd',
                        help = 'weight decay',
                        default = 0.0005, type = float)
    parser.add_argument('--batch', dest = 'batch_size',
                        help = 'bach size',
                        default = 1, type = int)

    parser.add_argument('--log', dest = 'log_file',
                        help = 'path for saving loss record',
                        default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/lib/log_for_first_time.txt', type = str)
    parser.add_argument('--snapshot', dest = 'snapshot',
                        help = 'save model every snapshot iters',
                        default = 3000, type = int)
    parser.add_argument('--output', dest = 'output',
                        help = 'path for snapshot',
                        default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/snapshot', type = str)
    parser.add_argument('--neg_rate', dest = 'neg_rate',
                        help = 'the ratio of negVox to posVox',
                        default = 0, type = int)

    if len(sys.argv) == 1:

        parser.print_help()
#        sys.exit(1)
    args = parser.parse_args()
    return args
#----------------------weight initialization--------------------------------------
def weight_init(net):
    for m in net.modules():
        m.name = m.__class__.__name__
        if m.name.find('Conv')!=-1:
            nn.init.normal(m.weight.data, std = 0.01)
            if m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)

#-----------------------learning rate update policy--------------------------------
def poly_scheduler(optimizer, iteration, args):
    #update learning rate with poly policy
    power = args.power
    max_iters = args.max_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*((1.0 - float(iteration)/float(max_iters))**power)    
#------------------------train_net------------------------------------------------
def train_net(net, train_data, optimizer, args):
    max_iters = args.max_iters
    data_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 1)
     
    #set phase
    net.train(True)

    loss_weight = [5.0, 1.0, 2.0]
    current_iter = 0
    while current_iter < max_iters:
        for iteration, (input_data, target) in enumerate(data_loader):
            ''' if not current_iter%10: 
                logging.info('Iteration {}, lr = {:.8f}'.format(current_iter, optimizer.param_groups[0]['lr']))
            if args.use_gpu:
                input_data = input_data.cuda()
                target = target.cuda()
            
            input_data, target = Variable(input_data), Variable(target)

            if DEBUG:
                visualization(input_data, target)

            optimizer.zero_grad()
            out = net.forward(input_data)
            loss_seq = []
            target = target.view(target.numel(), -1).squeeze()
            for i, o in enumerate(out):
                loss_seq.append(nnf.nll_loss(o, target, ignore_index = 2)*loss_weight[i])
                grad_seq = [loss_seq[0].data.new(1).fill_(1) for _ in range(len(loss_seq))]
            torch.autograd.backward(loss_seq, grad_seq)  
            if not current_iter%10:
                logging.info('Iteration {}, main_loss = {}, aux1_loss = {}, aux2_loss = {}'.format(current_iter, 
                             loss_seq[0].data.cpu().numpy(), loss_seq[1].data.cpu().numpy(), loss_seq[2].data.cpu().numpy()))
            poly_scheduler(optimizer, iteration, args)
            
            if not current_iter%10:
                logging.info('after optimizer')
 
            if not (current_iter+1)%args.snapshot:
                torch.save(net, os.path.join(args.output, 'model_iter_' + '{}'.format(iteration) + '.pkl'))
                torch.save(net.state_dict(), os.path.join(args.output, 'params_iter_' + '{}'.format(iteration) + '.pkl'))
            current_iter += 1 '''
            if DEBUG:
                target = target.view(target.numel(), -1).squeeze()
                logging.info('Iteration {}, lr = {:.8f}'.format(current_iter, optimizer.param_groups[0]['lr']))
                
                if args.use_gpu:
                    input_data = input_data.cuda()
                    target = target.cuda()
                input_data, target = Variable(input_data), Variable(target)
                optimizer.zero_grad()
                out = net(input_data)
                poly_scheduler(optimizer, iteration, args)
                loss = nnf.nll_loss(out, target, ignore_index = 2)
                loss.backward()
                logging.info('Iteration {}, aux1_loss = {}'.format(current_iter, loss.data.cpu().numpy()))
                current_iter += 1

    return net
#------------------------main: train the net--------------------------------------- 
if __name__ == '__main__':
    
    #parse the arguments 
    args = parse_args() 
    #initial log file
    logging.basicConfig(level = logging.DEBUG,
            format = '%(asctime)s [%(levelname)s] at %(filename)s, %(message)s',
            datefmt = '%Y-%m-%d(%a)%H:%M:%S', filename = args.log_file, filemode = 'w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)-3s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
   
    logging.info( 'use_gpu: {}'.format(args.use_gpu))
    logging.info( 'gpu_id: {}'.format(args.gpu_id))
    logging.info( 'max_iter: {}'.format(args.max_iters))
    logging.info( 'pretrained_model: {}'.format(args.pretrained_model))
    logging.info( 'data_path: {}'.format(args.data_path))
    logging.info( 'power: {}'.format(args.power))
    logging.info( 'learning_rate: {}'.format(args.base_lr))
    logging.info( 'weight_decay: {}'.format(args.base_wd))
    logging.info( 'batch_size: {}'.format(args.batch_size))
    logging.info( 'neg_rate: {}'.format(args.neg_rate))
    logging.info( 'log_file: {}'.format(args.log_file))
    logging.info( 'snapshot: {}'.format(args.snapshot))
    logging.info( 'output: {}'.format(args.output))

    #prepare net struvture and traing strategy    
    net = DSN_net()
    if not torch.cuda.is_available():
        logging.warn('there is no gpu available')
    
    net.apply(weight_init)
    
    if args.use_gpu:
        net = net.cuda()  

#    g = make_dot(main, params = dict(net.named_parameters()))
#    g.view()
    para_set_list = []


    #lr_mult and wd_mult for different layers
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            para_set_list.append({'params': m.weight, 'lr': args.base_lr*1.0, 'weight_decay': args.base_wd*1.0})
            para_set_list.append({'params': m.bias, 'lr': args.base_lr*2.0, 'weight_decay': 0.0})
    para_set_list.append({'params': net.deconv_aux1a.weight, 'lr': args.base_lr*0.1, 
                          'weight_decay': args.base_wd*1.0})
    para_set_list.append({'params': net.deconv_aux2a.weight, 'lr': args.base_lr*0.1,
                          'weight_decay': args.base_wd*1.0})
    para_set_list.append({'params': net.deconv_aux2b.weight, 'lr': args.base_lr*0.1, 
                          'weight_decay': args.base_wd*1.0})
    optimizer = optim.SGD(para_set_list, lr = args.base_lr, momentum = 0.9, weight_decay = args.base_wd)

    #prepare training data
    img_path = os.path.join(args.data_path, 'h5_data')
    txt_path = os.path.join(args.data_path, 'train.list')
    train_data = data.MyDataset(img_path, txt_path, 'rd_transp', args)
    
    #train net
    net = train_net(net, train_data, optimizer, args)

#    loss = criterion_main(main, target)*5.0 + criterion_aux1(aux1, target)*1.0 + criterion_aux2(aux2, target)*2.0
#    loss_seq = []
#    loss_weight = [5.0, 1.0, 2.0]

#    for i, o in enumerate(out):
#        loss_seq.append(nnf.nll_loss(o, target)*loss_weight[i])
#    torch.autograd.backward(loss_seq, grad_seq)   
#    for param_group in optimizer.param_groups:
#        print param_group['lr']






