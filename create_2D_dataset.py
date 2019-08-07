import os, math
import numpy as np
import cv2
from get_data_list import get_data_list

data_path = './Data/BRATS2015_Training/HGG'
validation_ratio = 0.2

Data = get_data_list(data_path)

num_examples = Data.len

file_indices = 'indexes.npy'

if os.path.isfile(file_indices):
    sample_indexes = np.load(file_indices)
    print('Random list of samples LOADED')

split_index = int(math.ceil(num_examples * (1 - validation_ratio)))

train_indices = sample_indexes[0: split_index]
print(train_indices)

validation_indices = sample_indexes[split_index:]
print(validation_indices)

t_f_gt = []
t_f_flair = []
for index in train_indices:
    t_f_gt.append(Data.GT[index])
    t_f_flair.append(Data.FLAIR[index])
v_f_gt = []
v_f_flair = []
for index in validation_indices:
    v_f_gt.append(Data.GT[index])
    v_f_flair.append(Data.FLAIR[index])

t_f = zip(t_f_flair, t_f_gt)
v_f =  zip(v_f_flair, v_f_gt)

for vol_name, gt_name in t_f:
    vol = np.load(vol_name)
    gt = np.load(gt_name)
    name = vol_name.split('/')[4]

    for i in range(155):
        im = vol[:,:,i]
        gt2d = gt[:,:,i]
        if not(np.sum(np.reshape(im,newshape=[-1])) == 0):
            np.save('./Data2D/train/im_'+name+'_'+str(i)+'.npy', im)
            # norm_im = im-np.min(np.min(im))
            # norm_im = norm_im/np.max(np.max(norm_im))
            # cv2.imwrite('./Data2D/train/im_'+name+'_'+str(i)+'.png', 255*norm_im)
            np.save('./Data2D/train/gt_'+name+'_'+str(i)+'.npy', gt2d)


for vol_name, gt_name in v_f:
    vol = np.load(vol_name)
    gt = np.load(gt_name)
    name = vol_name.split('/')[4]

    for i in range(155):
        im = vol[:, :, i]
        gt2d = gt[:, :, i]
        if not (np.sum(np.reshape(im, newshape=[-1])) == 0):
            np.save('./Data2D/validation/im_' + name + '_' + str(i) + '.npy', im)
            np.save('./Data2D/validation/gt_' + name + '_' + str(i) + '.npy', gt2d)
