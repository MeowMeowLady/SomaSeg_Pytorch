from torch.utils.data import Dataset
from PIL import Image
import h5py as hf
import os
import numpy as np
import numpy.random as rand
import random
import torch
import logging
import pdb
DEBUG = False

class h5dataset(Dataset):
    def __init__(self, img_path, txt_path, args, data_transform = None):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.rstrip().split('/')[-1]) for line in lines]
        self.data_transform = data_transform
        self.neg_rate = args.neg_rate
        self.patch_size = args.patch_size

    def __len__(self):
        return len(self.img_name)

    
    # if need, add dont care are for images and labels
    def add_dtcare(self, label, neg_rate):
        label_flat = np.reshape(label, (label.size, -1))
        posVox = np.nonzero(label_flat == 1)
        negVox = np.nonzero(label_flat == 0)# tuple has 2 1-d arrays
        negTot = len(negVox[0])
        posTot = len(posVox[0])
     #   label_flat[0,0] = 100
     #   print label_flat[0,0], label[0,0,0]
     #   pdb.set_trace()
     #   print 'neg before add dtcare: {}'.format(negTot)
        if posTot == 0:
           dtcareNew = negTot - int(negTot*0.05)
           dtcareIdx = random.sample(range(negTot), dtcareNew)
      #     print 'dtcare number to add: {}'.format(len(dtcareIdx))
           label_flat[negVox[0][dtcareIdx]] = 2
        else:
            if DEBUG:
               print 'pos: {}'.format(posTot)
            if float(negTot)/float(posTot) > neg_rate:
                dtcareNew = negTot - posTot*neg_rate
                dtcareIdx = random.sample(range(negTot), dtcareNew)
       #         print 'dtcare number to add: {}'.format(len(dtcareIdx))
                label_flat[negVox[0][dtcareIdx]] = 2
        label = np.reshape(label_flat, label.shape)
        #label_flat2 = np.reshape(label, (label.size, -1))
        #posVox = np.nonzero(label_flat2 == 1)
        #negVox = np.nonzero(label_flat2 == 0)# tuple has 2 1-d arrays
        #negTot = len(negVox[0])
        #posTot = len(posVox[0])
        #print 'after add dtcare: pos = {}, neg = {}'.format(posTot, negTot)
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

    # randomly crop image cubes to smaller ones
    def rd_crop(self, img, label):
        ss = self.patch_size
        [d1, d2, d3] = img.shape

        if ss[0] > d1 or ss[1] > d2 or ss[2] > d3:
            logging.error('the patch_size must be smaller than the original size')
            os._exit(0)
        from_d1 = rand.randint(d1 - ss[0] + 1)
        from_d2 = rand.randint(d2 - ss[1] + 1)
        from_d3 = rand.randint(d3 - ss[2] + 1)
        cropped_img = img[from_d1: from_d1+ss[0], from_d2: from_d2+ss[1], from_d3: from_d3+ss[2]]
        cropped_label = label[from_d1: from_d1+ss[0], from_d2: from_d2+ss[1], from_d3: from_d3+ss[2]]
        
        return cropped_img, cropped_label

    #pre-process normalization  img:[S, H, W]/slice, height, width
    def pre_process(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        img = (img - mean_val)/std_val
        return img

   

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img_fid = hf.File(img_name)
        img = img_fid['data'][:]
        label = img_fid['label'][:]
        img = np.squeeze(img)
        label = np.squeeze(label)
        #crop images
        if not img.shape == self.patch_size:
            cropped_img, cropped_label = self.rd_crop(img, label)
        #normalization /pro-press
        norm_cropped_img = self.pre_process(cropped_img)
        

        if DEBUG:
            logging.info('before add_dtcare')
        if self.neg_rate:
            cropped_label = self.add_dtcare(cropped_label, self.neg_rate)
        
        if DEBUG:
            logging.info('after add_dtcare)')
        if self.data_transform is not None:
            try:
                if self.data_transform == 'transp':
                    norm_cropped_img, cropped_label = self.rd_transp(norm_cropped_img, cropped_label)
            except:
                print("Cannot transform image: {}".format(img_name))
        if DEBUG:
            logging.info('after transform')
        norm_cropped_img = norm_cropped_img[np.newaxis, :]
        cropped_label = cropped_label[np.newaxis, :]
        img_tensor = torch.from_numpy(norm_cropped_img).float()
	label_tensor = torch.from_numpy(cropped_label).long()#float will return error, don't konw why
#        print img_tensor.shape, label_tensor.shape
        return img_tensor, label_tensor
      
