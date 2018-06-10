import numpy as np
import cv2
from dsn import DSN as Net
import os
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
import torch.nn.functional as nnf
import argparse
import sys
import logging
import scipy.io as scio
import h5py
import time
from libtiff import TIFF
import gc
DEBUG = False

#--------------------configurations------------------------------------------------
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test DSN')
    parser.add_argument('--gpu', dest = 'gpu_id', help = 'GPU device id to use [0]', default = 0, type = int)
    parser.add_argument('--use_gpu', dest = 'use_gpu', help = 'whether use gpu', default = True, type = bool)
    parser.add_argument('--weight', dest = 'weight', help = 'the path for pretrained model', 
                        default = '/media/dongmeng/Data/Code/SomaSeg_pytorch/snapshot/20180501/model_epoch_10.pkl', type = str)
    parser.add_argument('--data', dest = 'test_data', help = 'the path for test data', default = '/media/dongmeng/Meng/fullres', type = str)
    parser.add_argument('--result', dest = 'result', help = 'the path for saving test results', default = '/media/dongmeng/Hulk/fullres/result', type = str)
    parser.add_argument('--patch_size', dest = 'patch_size', help = 'the input size for net', default = [64, 240, 240], type = int)
    parser.add_argument('--overlap', dest = 'overlap', help = 'the overlap between cropped cubes', default = 20, type = int)
    parser.add_argument('--log', dest = 'log', help = 'log file name', default = 'ptlog_demo.txt', type = str)
    

    if len(sys.argv) == 1:

        parser.print_help()
#        sys.exit(1)
    args = parser.parse_args()
    return args
#---------------------pre_process--------------------------------------------------   
#pre-process normalization  img:[S, H, W]/slice, height, width
def pre_process(img):
    #img = img.transpose(2, 1, 0)
    img = np.array(img, dtype = 'float')
    mean_val = np.mean(img)
    std_val = np.std(img)
    if std_val == 0:
        img = img - mean_val
        return img
    img = (img - mean_val)/std_val
    return img
#------------------------generate_score_map------------------------------------------
 
def generate_score_map(net, img, args):
    patch_size = args.patch_size
    overlap = args.overlap

    slices, height, width = img.shape
    
    #the score map for forground and background
    score_map = np.zeros((slices, height, width, 2), dtype = 'float16')
    #count the times of adding score (for average)
    cnt = np.zeros((slices, height, width), dtype = 'float16')

    sidx = range(0, slices - patch_size[0], patch_size[0] - overlap) + [slices - patch_size[0]]
    hidx = range(0, height - patch_size[1], patch_size[1] - overlap) + [height - patch_size[1]]
    widx = range(0, width - patch_size[2], patch_size[2] - overlap) + [width - patch_size[2]]
    for ss in sidx:
        for hs in hidx:
            for ws in widx:
                input_cube = img[ss: ss+patch_size[0], hs: hs+patch_size[1], ws: ws+patch_size[2]]
                #minus mean val and divive var val
                input_cube = pre_process(input_cube)

                input_cube = input_cube[np.newaxis, np.newaxis, :]
                if DEBUG:
                    print 'imput cube size{}'.format(input_cube.shape)
                input_data = Variable(torch.from_numpy(input_cube).float())
                if args.use_gpu:
                    input_data = input_data.cuda()
                output = net(input_data)
                pred = nnf.softmax(output[0], dim = 1)
                pred = pred.data.cpu().numpy()

                pred = np.reshape(pred, (patch_size[0], patch_size[1], patch_size[2], 2))
                score_map[ss: ss+patch_size[0], hs: hs+patch_size[1], ws: ws+patch_size[2], :] += pred
                cnt[ss: ss+patch_size[0], hs: hs+patch_size[1], ws: ws+patch_size[2]] += 1.0
    del img
    gc.collect()
    cnt = cnt[:, :, :, np.newaxis]
    cnt = np.concatenate((cnt, cnt), axis = 3)
    score_map = score_map/cnt
    del cnt
    gc.collect()
    return score_map
#----------------------------sort_img_path_by_name------------------------------------
def sort_by_name(file_name):

    number_list = []
    for name in file_name:
        number_list.append(int(name.split('_')[0]))
    sorted_idx = np.argsort(number_list)
    sorted_file_name = [file_name[x] for x in sorted_idx]
    return sorted_file_name
    
def rm_small_connected_component(img):
    return


if __name__ == '__main__':
    #parse the arguments
    args = parse_args()
    
    #check if the result path exist
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    #initial log filei
    time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level = logging.DEBUG,
            format = '%(asctime)s [%(levelname)s] at %(filename)s, %(message)s',
            datefmt = '%Y-%m-%d(%a)%H:%M:%S', filename = os.path.join(args.result, 'ptlog_demo'+ time_now +'.txt'), filemode = 'w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info( 'use_gpu: {}'.format(args.use_gpu))
    logging.info( 'gpu_id: {}'.format(args.gpu_id))
    logging.info( 'weight: {}'.format(args.weight))
    logging.info( 'test_data: {}'.format(args.test_data))
    logging.info( 'patchsize: {}'.format(args.patch_size))
    logging.info( 'result: {}'.format(args.result))
    logging.info( 'overlap: {}'.format(args.overlap))
    logging.info( 'log: {}'.format(args.log))

    #set cudnn    
    cudnn.benchmark = True 
    logging.info('start loading net...')    
    #prepare net and load model
    net = Net()
    net = torch.load(args.weight)
    if args.use_gpu:
        net = net.cuda()
    net.eval()
    logging.info('done!')
        
    #read image 
    '''
    img_list = [] 
    img_path = args.test_data
    for root, dirs, files in os.walk(img_path):
        for im_file in files:
    	    if os.path.splitext(im_file)[1] == '.tif' and '_' in im_file:
    	        img_list.append(im_file)
    
    #get the sequential image path list
    img_list = sort_by_name(img_list)
    
    if img_list is None:
        logging.error('no image is available!')
        exit(0)
    if len(img_list) == 1:
        logging.error('no 3d image is available!')
        exit(0)
    
    img = cv2.imread(os.path.join(img_path, img_list[0]), -1)
    img = img[np.newaxis, :]
    logging.info ('start reading images...')
    for img_name in img_list[1:]:
        img_read = cv2.imread(os.path.join(img_path, img_name), -1)
        img_read = img_read[np.newaxis, :]
        img = np.concatenate((img, img_read), axis = 0)
    logging.info('done!')
    '''
    for i in [0, 8, 9]:
        if i == 8:
            l = [8, 9]
        else:
            l = range(10)
        for j in l:
            logging.info ('start reading images {}_{}.tif...'.format(i, j))
            img_list = []
            tif = TIFF.open(os.path.join(args.test_data, '{}_{}.tif'.format(i, j)), mode='r')
            img = np.zeros((1253, 2187, 2557), dtype = 'uint16')
            totalIm = 0
            for image in tif.iter_images():
                img[totalIm, :, :] = tif.read_image()
                totalIm += 1
                #img_list.append(img)
            #img = np.array(img_list)
            #del img_list
            gc.collect() 
            logging.info('done!')
            
            logging.info('start testing...')
            start = [0, 626]
            end = [626, 1253]
            for half in [0, 1]:
                #generate smaller cubes to fit the input size of net
                img_half = img[start[half]: end[half], :, :]
                score_map = generate_score_map(net, img_half, args)
    
                #save score map
                np.save(os.path.join(args.result, '{}_{}_{}.npy'.format(i, j, half)), score_map[:,:,:,1])
                del score_map
                del img_half
                gc.collect()
                logging.info('{}_{}_{} done!'.format(i, j, half))
       
    


        

