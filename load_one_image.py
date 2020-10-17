import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import shutil
from tqdm import tqdm
import cv2

datapath = './Data_npy/test/'
datapath_dest = './Data/test/'

# datapath = './Data_npy/train'
# datapath_dest = './Data/train/'

onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
gt_list = [f for f in onlyfiles if 'gt' in f]
im_list = [f for f in onlyfiles if 'im' in f]

# for f in tqdm(gt_list):
#     gt = np.load(datapath+'/'+f)
#     gt = 255.*gt[:,:, np.newaxis]
#
#     cv2.imwrite(datapath_dest+'/'+f[:-3]+'png', gt)

for f in tqdm(im_list):
    im = 255.*np.load(datapath + '/' + f)
    if not(np.sum(im)==0):
        cv2.imwrite(datapath_dest+'/'+f[:-3]+'png', im)

        gt = np.load(datapath + '/' + 'gt'+f[2:])
        gt = 255. * gt[:, :, np.newaxis]

        cv2.imwrite(datapath_dest + '/gt' + f[2:-3] + 'png', gt)


    # plt.imshow(im)
    # plt.show()
    # print(2)

# print(onlyfiles)