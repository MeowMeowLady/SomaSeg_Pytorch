from torch.utils.data import Dataset
from PIL import Image
import h5py as hf
import os
import numpy as np
import random
import torch

DEBUG = False
class MyDataset(Dataset):
    def __init__(self, img_path, txt_path, data_transform, args):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.rstrip().split('/')[-1]) for line in lines]
        self.data_transform = data_transform
        self.neg_rate = args.neg_rate

    def __len__(self):
        return len(self.img_name)

    
    # if need, add dont care are for images and labels
    def add_dtcare(iself, label, neg_rate):
        label_flat = np.reshape(label, (label.size, -1))
        posVox = np.nonzero(label_flat == 1)
        negVox = np.nonzero(label_flat == 0)# tuple has 2 1-d arrays
        negTot = len(negVox[0])
        posTot = len(posVox[0])
        if float(negTot)/float(posTot) > neg_rate:
            dtcareNew = negTot - posTot*neg_rate
            dtcareIdx = random.sample(range(negTot), dtcareNew)
            #label_flat = np.reshape(label, (label.size, -1)) 
        print 'here'
             # label_flat[negVox[0][x]][0] = 2 for 
        return label

    # randomly transpose the imput data
    def rd_transp(self, img, label):
        prob = random.rand()
        if prob > 0.5:
            if prob > 0.75:
                trans_img = np.transpose(img, (1, 2, 0))
                trans_label = np.transpose(label, (1, 2, 0))
            else:
                trans_img = np.transpose(img, (2, 0, 1))
                trans_label = np.transpose(label, (2, 0, 1))
        else:
            trans_img = img
            trans_label = label
        return trans_img, trans_label 

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img_fid = hf.File(img_name)
        img = img_fid['data'][:]
        label = img_fid['label'][:]
        if DEBUG:
            print type(img)
            print 'origional:{}, {}'.format(img.shape, label.shape)
        img = np.squeeze(img)
        label = np.squeeze(label)
        
        if self.neg_rate:
            label = self.add_dtcare(label, self.neg_rate)
        
        if self.data_transform is not None:
            try:
                if self.data_transform == 'transp':
                    img, label = self.rd_transp(img, label)
            except:
                print("Cannot transform image: {}".format(img_name))
        img = img[np.newaxis, :]
        label = label[np.newaxis, :]
        img_tensor = torch.from_numpy(img).float()
	label_tensor = torch.from_numpy(label).long()
#        print img_tensor.shape, label_tensor.shape
        return img_tensor, label_tensor
      
