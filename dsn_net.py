from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import argparse
import sys
import logging
import MyDataset
from Visualize import make_dot
import torch
import numpy as np
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
        self.conv4b = nn.Conv3d(256, 256, 3, 1, 1, bias = True)
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
        aux1 = nnf.log_softmax(aux1)
        main = self.pool2(main)
        main = nnf.relu(self.bn3a(self.conv3a(main)))
        main = nnf.relu(self.bn3b(self.conv3b(main)))
        aux2 = self.score_aux2(self.deconv_aux2b(self.deconv_aux2a(main)))
        #make channels the first axis 
        aux2 = aux2.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        aux2 = aux2.view(aux2.numel()//2, 2)
        aux2 = nnf.log_softmax(aux2)
        main = self.pool3(main)
        main = nnf.relu(self.bn4a(self.conv4a(main)))
        main = nnf.relu(self.bn4b(self.conv4b(main)))
        main = self.deconv3a(self.deconv2a(self.deconv1a(main)))
        main = self.score_main(main)
        #make channels the first axis 
        main = main.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        main = main.view(main.numel()//2, 2)
        main = nnf.log_softmax(main)
        return main, aux1, aux2

#----------------------weight initialization--------------------------------------
def weights_init(net):
    for m in net.modules():
        m.name = m.__class__.__name__
        if m.name.find('Conv')!=-1:
            nn.init.norml(m.weight.data, std = 0.01)
            nn.init.constant(m.bias.data, 0.0)
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
                        default='/media/dongmeng/Data/Code/SomaSeg/data/', type=str)
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
                        default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/lib/log.txt', type = str)
    parser.add_argument('--snapshot', dest = 'snapshot',
                        help = 'save model every snapshot iters',
                        default = 3000, type = int)
    parser.add_argument('--output', dest = 'output',
                        help = 'path for snapshot',
                        default = '', type = str)
#    parser.add_argument('--rand', dest='randomize',
#                        help='randomize (do not use a fixed seed)',
#                        action='store_true')

    if len(sys.argv) == 1:

        parser.print_help()
#        sys.exit(1)
    args = parser.parse_args()
    return args

#-----------------------learning rate update policy--------------------------------
def adjust_learning_rate(optimizer, iter, args):
    #update learning rate with poly policy
    power = args.power
    max_iters = args.max_iters
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*((1.0 - float(iter)/float(max_iters))**power)    

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
   
    logging.info( 'gpu_id: {}'.format(args.gpu_id))
    logging.info( 'max_iter: {}'.format(args.max_iters))
    logging.info( 'pretrained_model: {}'.format(args.pretrained_model))
    logging.info( 'data_path: {}'.format(args.data_path))
    logging.info( 'power: {}'.format(args.power))
    logging.info( 'learning_rate: {}'.format(args.base_lr))
    logging.info( 'weight_decay: {}'.format(args.base_wd))
    logging.info( 'batch_size: {}'.format(args.batch_size))
    logging.info( 'log_file: {}'.format(args.log_file))
    logging.info( 'snapshot: {}'.format(args.snapshot))
    logging.info( 'output: {}'.format(args.output))

    #prepare net struvture and traing strategy    
    net = DSN_net()
    
#    g = make_dot(main, params = dict(net.named_parameters()))
#    g.view()
   

#    criterion_main = nnf.nll_loss()
#    criterion_aux1 = nnf.nll_loss()
#    criterion_aux2 = nnf.nll_loss()
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
    x = Variable(torch.randn(1, 1, 160, 160, 160))
    target = Variable(torch.from_numpy(np.ones((1, 1, 160, 160, 160)).astype(np.int32)).long())
 #   target = Variable(torch.LongTensor(1, 1, 160, 160, 160)_zeros)
    target = target.permute(0, 2, 3, 4, 1).contiguous()
    #flatten
    target = target.view(1, target.numel())
    target = torch.squeeze(target)
    net = DSN_net()
    out = net.forward(x)
#    loss = criterion_main(main, target)*5.0 + criterion_aux1(aux1, target)*1.0 + criterion_aux2(aux2, target)*2.0
    loss_seq = []
    loss_weight = [5.0, 1.0, 2.0]
    for i, o in enumerate(out):
        loss_seq.append(nnf.nll_loss(o, target)*loss_weight[i])
    grad_seq = [loss_seq[0].data.new(1).fill_(1) for _ in range(len(loss_seq))]
    torch.autograd.backward(loss_seq, grad_seq)    
    
    max_iter = args.max_iters
    for param_group in optimizer.param_groups:
        print param_group['lr'] 

    adjust_learning_rate(optimizer, 1, args)

    out = net.forward(x)
#    loss = criterion_main(main, target)*5.0 + criterion_aux1(aux1, target)*1.0 + criterion_aux2(aux2, target)*2.0
    loss_seq = []
    loss_weight = [5.0, 1.0, 2.0]
    for i, o in enumerate(out):
        loss_seq.append(nnf.nll_loss(o, target)*loss_weight[i])
    torch.autograd.backward(loss_seq, grad_seq)   
    for param_group in optimizer.param_groups:
        print param_group['lr']






