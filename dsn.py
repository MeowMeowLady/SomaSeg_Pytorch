import torch
import torch.nn as nn
import torch.nn.functional as nnf

class DSN(nn.Module):
    def __init__(self):
        super(DSN, self).__init__()
        self.conv1a = nn.Conv3d(1, 32, 3, 1, 1, bias = True)
        self.bn1a=nn.BatchNorm3d(32, momentum = 0.001, affine=True)
        self.pool1 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv2a = nn.Conv3d(32, 64, 3, 1, 1, bias = True)
        self.bn2a = nn.BatchNorm3d(64, momentum = 0.001, affine = True)
        self.conv2b = nn.Conv3d(64, 64, 3, 1, 1, bias = True)
        self.bn2b = nn.BatchNorm3d(64, momentum = 0.001, affine = True)
        self.pool2 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv3a = nn.Conv3d(64, 128, 3, 1, 1, bias = True)
        self.bn3a = nn.BatchNorm3d(128, momentum = 0.001, affine = True)
        self.conv3b = nn.Conv3d(128, 128, 3, 1, 1, bias = True)
        self.bn3b = nn.BatchNorm3d(128, momentum = 0.001, affine = True)
        self.pool3 = nn.MaxPool3d(2, 2, padding = 0)
        self.conv4a = nn.Conv3d(128, 256, 3, 1, 1, bias = True)
        self.bn4a = nn.BatchNorm3d(256, momentum = 0.001, affine = True)
        self.conv4b = nn.Conv3d(256, 256, 3, 1, 1, bias = True)
        self.bn4b = nn.BatchNorm3d(256, momentum = 0.001, affine = True)
        self.deconv1a = nn.ConvTranspose3d(256, 128, 4, 2, 1, bias = True)
        self.deconv2a = nn.ConvTranspose3d(128, 64, 4, 2, 1, bias = True)
        self.deconv3a = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = True)
        self.score_main = nn.Conv3d(32, 2, 1, 1, bias = True)
        self.deconv_aux1a = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = False)
        self.score_aux1 = nn.Conv3d(32, 2, 1, bias = True)
        self.deconv_aux2a = nn.ConvTranspose3d(128, 64, 4, 2, 1, bias = False)
        self.deconv_aux2b = nn.ConvTranspose3d(64, 32, 4, 2, 1, bias = False)
        self.score_aux2 = nn.Conv3d(32, 2, 1, bias = True)

        self.__weight_init()

    #weight initialization
    def __weight_init(self):
        for m in self.modules():
            m.name = m.__class__.__name__
            if m.name.find('Conv')!=-1:
                nn.init.normal(m.weight.data, std = 0.01)
                if m.bias is not None:
                    nn.init.constant(m.bias.data, 0.0)
            if m.name.find('BatchNorm3d')!=-1:
                nn.init.constant(m.weight.data, 1.0)
                nn.init.constant(m.bias.data, 0.0)



    def forward(self, main):
        main = self.pool1(nnf.relu(self.bn1a(self.conv1a(main))))
        main = nnf.relu(self.bn2a(self.conv2a(main)))
        main = nnf.relu(self.bn2b(self.conv2b(main)))
        aux1 = self.score_aux1(self.deconv_aux1a(main))
        #make channels the first axis 
        aux1 = aux1.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        aux1 = aux1.view(aux1.numel()//2, 2)
        main = self.pool2(main)
        main = nnf.relu(self.bn3a(self.conv3a(main)))
        main = nnf.relu(self.bn3b(self.conv3b(main)))
        aux2 = self.score_aux2(self.deconv_aux2b(self.deconv_aux2a(main)))
        #make channels the first axis 
        aux2 = aux2.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        aux2 = aux2.view(aux2.numel()//2, 2)
#        aux2 = nnf.log_softmax(aux2, dim=1)
        main = self.pool3(main)
        main = nnf.relu(self.bn4a(self.conv4a(main)))
        main = nnf.relu(self.bn4b(self.conv4b(main)))
        main = self.deconv3a(self.deconv2a(self.deconv1a(main)))
        main = self.score_main(main)
        #make channels the first axis 
        main = main.permute(0, 2, 3, 4, 1).contiguous()
        #flatten
        main = main.view(main.numel()//2, 2)
#        main = nnf.log_softmax(main, dim=1)
        return main, aux1, aux2



    #-----------------------criterion-------------------------------------------------
    def crit_cross_entropy3d(self, out_seq, target, loss_weight):
        #compute loss for multi-output layers with different loss weight
        target = target.view(target.numel(), -1).squeeze()
        loss_seq = []
        grad_seq = []
        for i, o in enumerate(out_seq):
            loss_seq.append(nnf.nll_loss(nnf.log_softmax(o, dim = 1), target, ignore_index = 2)*loss_weight[i])
            grad_seq.append(loss_seq[i].data.new(1).fill_(1))

        return loss_seq, grad_seq

