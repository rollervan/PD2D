# coding=utf-8
import tensorflow as tf
import numpy as np
import time
from random import randint
import matplotlib.pyplot as plt
# import cv2
import os, shutil, re
import scipy.stats as st
import math
# import sklearn as sk
import scipy.io as sio
import scipy.ndimage.interpolation
from random import shuffle
from tqdm import tqdm
from optimizer_opt import optimizer_opt
from metrics import dice_coe, iou_coe
from PD_algorithm import PD
from u_net import unet
from ResNet_module import res_conv
from get_data_list import get_data_list
import random
from medpy.io import load

class PDNN:
    def __init__(self):
        self.globalpath = './'

        self.restore = False
        self.clear_logs = not(self.restore)
        self.is_test = True
        self.pretrain = False
        self.show_gradients = False

        self.model_name = 'PD2D'

        self.checkpoint = self.globalpath + 'Models/Model_' + self.model_name + '/model.ckpt'
        self.data_dir = self.globalpath + 'Data2D/'
        self.log_dir = self.globalpath + 'logs/' + self.model_name

        self.IM_ROWS = 240
        self.IM_COLS = 240
        self.IM_DEPTH = 1

        self.batch_norm = True
        self.eps = 1e-5
        self.learning_rate = 1e-5
        self.batch_size = 8
        self.epoch = 2000
        self.num_samples = 27280
        self.num_test_samples = 6820
        self.epoch_iteration = np.round( self.num_samples/(self.batch_size)).astype(int)
        self.test_each_epoch = 10

        # Print stuff
        print('batch_size',self.batch_size)
        print('epoch',self.epoch)
        print('epoch_iteration',self.epoch_iteration)
        print('test_each_epoch',self.test_each_epoch)

        if self.restore == False:
            # Remove previous logs
            if self.clear_logs:
                self.remove_logs(self.log_dir)

        if not os.path.exists('Models/Model_' + self.model_name):
            os.makedirs('Models/Model_' + self.model_name)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.training_iters = np.round( self.epoch*self.num_samples/(self.batch_size)).astype(int)
        print(self.training_iters)

    def remove_logs(self,folder):
        if os.path.exists(folder):
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

    def _tf_fspecial_gauss(self,size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def tf_ssim(self,img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

    def image_to_4d(self,image):
        image = tf.expand_dims(image, 0)
        image = tf.expand_dims(image, -1)
        return image

    def gkern(self,kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = np.asarray(kernel,dtype=np.float32)
        return kernel
    def soft_obstacles(self,x, alpha= 0.1):
        # return 1./(1. + tf.exp(-alpha*(x-0.5)))
        return 2/(1 + tf.exp(-2*(x-0.5))) -0.5
    def gaussian(self,x,size,sigma):

        gaussian_k = self.gkern(kernlen=size,nsig=sigma)
        gaussian_k = tf.constant(gaussian_k)
        gaussian_k = tf.reshape(gaussian_k, [size, size, 1, 1])
        r = tf.nn.conv2d(x, gaussian_k, [1, 1, 1, 1], padding="SAME")
        return r
    def gradient(self, u):
        x_filter = tf.constant([[0., 0., 0.],
                                [0., -1., 0.],
                                [0., 1., 0.]])
        # x_filter = tf.reshape(x_filter,[1,3,3,1])
        # cero_3x3 = tf.constant([[0., 0., 0.],
        #                         [0., 0., 0.],
        #                         [0., 0., 0.]])
        # cero_3x3 = tf.reshape(cero_3x3, [3, 3, 1])
        # x_filter = tf.concat([cero_3x3,x_filter,cero_3x3],axis=2)
        x_filter = tf.reshape(x_filter, [3, 3, 1, 1])

        y_filter = tf.constant([[0., 0., 0.],
                                [0., -1., 1.],
                                [0., 0., 0.]])
        # y_filter = tf.reshape(y_filter,[1,3,3,1])
        #
        # y_filter = tf.concat([cero_3x3,y_filter,cero_3x3],axis=2)
        #
        y_filter = tf.reshape(y_filter, [3, 3, 1, 1])


        ux = tf.nn.conv2d(u, x_filter, strides=(1,1,1,1), padding="SAME")
        uy = tf.nn.conv2d(u, y_filter, strides=(1,1,1,1), padding="SAME")
        return ux, uy
    def divergence(self,ux,uy):
        x_filter = tf.constant([[0., -1., 0.],
                                [0., 1., 0.],
                                [0., 0., 0.]])
        x_filter = tf.reshape(x_filter, [3, 3, 1, 1])


        y_filter = tf.constant([[0., 0., 0.],
                                [-1., 1., 0.],
                                [0., 0., 0.]])
        y_filter = tf.reshape(y_filter, [3, 3, 1, 1])

        uxx = tf.nn.conv2d(ux, x_filter, strides=(1,1,1,1), padding="SAME")
        uyy = tf.nn.conv2d(uy, y_filter, strides=(1,1,1,1), padding="SAME")
        return uxx + uyy

    def TV2D(self,u):
        ux,uy = self.gradient(u)
        tv = tf.sqrt(tf.square(ux) + tf.square(uy) + 1e-6)
        return tv

    def _weight_variable(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))

    def _bias_variable(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))

    def peak_signal_to_noise_ratio(self,true, pred):
        """Image quality metric based on maximal signal power vs. power of the noise.
        Args:
          true: the ground truth image.
          pred: the predicted image.
        Returns:
          peak signal to noise ratio (PSNR)
        """
        return 10.0 * tf.log(1.0 / self.mean_squared_error(true, pred)) / tf.log(10.0)

    def mean_squared_error(self,true, pred):
        """L2 distance between tensors true and pred.
        Args:
          true: the ground truth image.
          pred: the predicted image.
        Returns:
          mean squared error between ground truth and predicted image.
        """
        return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))
    def dice(self, a, b):
        """
        Implementacion del coeficiente de dice en TensorFlow
        """
        return 2. * tf.reduce_sum(a * b) / (tf.reduce_sum(a) + tf.reduce_sum(b))

    def dice_coe(self, output, target, epsilon=0):
        """Sørensen–Dice coefficient for comparing the similarity of two distributions,
        usually be used for binary image segmentation i.e. labels are binary.
        The coefficient = [0, 1], 1 if totally match.
        Parameters
        -----------
        output : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        target : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        epsilon : float
            An optional name to attach to this layer.
        Examples
        ---------
        # >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
        # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_, epsilon=1e-5)
        References
        -----------
        - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
        """
        # inse = tf.reduce_sum( tf.mul(output, target) )
        # l = tf.reduce_sum( tf.mul(output, output) )
        # r = tf.reduce_sum( tf.mul(target, target) )
        inse = tf.reduce_sum(output * target)
        l = tf.reduce_sum(output * output)
        r = tf.reduce_sum(target * target)
        dice = 2 * (inse) / (l + r)
        if epsilon == 0:
            return dice
        else:
            return tf.clip_by_value(dice, 0, 1.0 - epsilon)
    def np_dice(self, output, target, epsilon=0):
        """Sørensen–Dice coefficient for comparing the similarity of two distributions,
        usually be used for binary image segmentation i.e. labels are binary.
        The coefficient = [0, 1], 1 if totally match.
        Parameters
        -----------
        output : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        target : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        epsilon : float
            An optional name to attach to this layer.
        Examples
        ---------
        # >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
        # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_, epsilon=1e-5)
        References
        -----------
        - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
        """
        # inse = tf.reduce_sum( tf.mul(output, target) )
        # l = tf.reduce_sum( tf.mul(output, output) )
        # r = tf.reduce_sum( tf.mul(target, target) )
        inse = np.sum(output * target)
        l = np.sum(output * output)
        r = np.sum(target * target)
        dice = 2 * (inse) / (l + r)
        if epsilon == 0:
            return dice
        else:
            return np.clip(dice, 0, 1.0 - epsilon)
    def np_iou(self, output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
        pre = (output > threshold).astype(np.float32)
        truth = (target > threshold).astype(np.float32)
        inse = np.sum(np.multiply(pre, truth))  # AND
        union = np.sum((np.add(pre, truth) >= 1).astype(np.float32))  # OR
        # old axis=[0,1,2,3]
        # epsilon = 1e-5
        # batch_iou = inse / (union + epsilon)
        # new haodong
        batch_iou = (inse + smooth) / (union + smooth)
        iou = np.mean(batch_iou)
        return iou  # , pre, truth, inse, union

    def create_list(self,data_path):

        file_indices = 'indexes2D.npy'

        if os.path.isfile(file_indices):
            [t_f, v_f] = np.load(file_indices,allow_pickle=True)
            print('Random list of samples LOADED')

        else:
            t_f_gt = []
            t_f_flair = []
            v_f_gt = []
            v_f_flair = []
            for root, dirs, files in os.walk(data_path+'/train'):
                for filename in files:
                    if filename[0:2] == 'gt':
                        t_f_gt.append(data_path+'/train/'+filename)
                        t_f_flair.append(data_path+'/train/im'+filename[2:])
            for root, dirs, files in os.walk(data_path+'/validation'):
                for filename in files:
                    if filename[0:2] == 'gt':
                        v_f_gt.append(data_path+'/validation/'+filename)
                        v_f_flair.append(data_path+'/validation/im'+filename[2:])

            t_f = [t_f_flair, t_f_gt]
            v_f = [v_f_flair, v_f_gt]

            np.save(file_indices,[t_f,v_f])
            print('Random list of samples CREATED')
            # print(sample_indexes)

        return t_f, v_f

    def load_data_train(self,data_path, list):
        num_images = len(list[0])

        list_flair = list[0]
        list_gt = list[1]

        # list_max = []
        # list_mean = []
        # list_std = []
        # for name in list_flair:
        #     im,_= load(name)
        #     list_max.append(np.max(im))
        #     list_mean.append(np.mean(im))
        #     list_std.append(np.std(im))
        # list_max= np.asarray(list_max)
        # list_mean= np.asarray(list_mean)
        # list_std= np.asarray(list_std)
        # plt.plot(list_max)
        # plt.plot(list_mean)
        # plt.plot(list_mean-3*list_std)
        # plt.plot(list_mean+3*list_std)
        # plt.legend('Max','Mean','Mean-3*STD','Mean+3*STD')
        i = 0
        while 1:

            im_list = []
            msk_list = []
            for ii in range(self.batch_size):
                im = np.load(list_flair[i+ii])
                # im = scipy.ndimage.interpolation.zoom(im,[2.,2.,2.],output=np.float32,order=0)
                #im = scipy.ndimage.interpolation.zoom(im, [0.5, 0.5, 0.5], output=np.float32, order=0)
                im_list.append(im)

                msk = np.load(list_gt[i+ii])
                # msk = np.maximum(0.,np.minimum(1.,10.*msk))
                # msk = scipy.ndimage.interpolation.zoom(msk,[2.,2.,2.],output=np.float32,order=0)
                #msk = scipy.ndimage.interpolation.zoom(msk, [0.5, 0.5, 0.5], output=np.float32, order=0)
                msk_list.append(msk)

                if i == num_images-self.batch_size:
                    i = 0
                else:
                    i = i + 1


            im = np.reshape(im_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            msk = np.reshape(msk_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            yield im, msk
    def load_data_valid(self,data_path, list):
        num_images = len(list[0])

        list_flair = list[0]
        list_gt = list[1]

        # list_max = []
        # list_mean = []
        # list_std = []
        # for name in list_flair:
        #     im,_= load(name)
        #     list_max.append(np.max(im))
        #     list_mean.append(np.mean(im))
        #     list_std.append(np.std(im))
        # list_max= np.asarray(list_max)
        # list_mean= np.asarray(list_mean)
        # list_std= np.asarray(list_std)
        # plt.plot(list_max)
        # plt.plot(list_mean)
        # plt.plot(list_mean-3*list_std)
        # plt.plot(list_mean+3*list_std)
        # plt.legend('Max','Mean','Mean-3*STD','Mean+3*STD')
        i = 0
        while 1:

            im_list = []
            msk_list = []
            for ii in range(self.batch_size):
                im = np.load(list_flair[i+ii])
                # im = scipy.ndimage.interpolation.zoom(im,[2.,2.,2.],output=np.float32,order=0)
                #im = scipy.ndimage.interpolation.zoom(im, [0.5, 0.5, 0.5], output=np.float32, order=0)
                im_list.append(im)

                msk = np.load(list_gt[i+ii])
                # msk = np.maximum(0.,np.minimum(1.,10.*msk))
                # msk = scipy.ndimage.interpolation.zoom(msk,[2.,2.,2.],output=np.float32,order=0)
                #msk = scipy.ndimage.interpolation.zoom(msk, [0.5, 0.5, 0.5], output=np.float32, order=0)
                msk_list.append(msk)

                if i == num_images-self.batch_size:
                    i = 0
                else:
                    i = i + 1


            im = np.reshape(im_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            msk = np.reshape(msk_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            yield im, msk
    def load_data_test(self,list):
        list = np.load('test_list.npy')
        num_images = len(list[0])

        list_flair = list[0]
        list_gt = list[1]

        # list_max = []
        # list_mean = []
        # list_std = []
        # for name in list_flair:
        #     im,_= load(name)
        #     list_max.append(np.max(im))
        #     list_mean.append(np.mean(im))
        #     list_std.append(np.std(im))
        # list_max= np.asarray(list_max)
        # list_mean= np.asarray(list_mean)
        # list_std= np.asarray(list_std)
        # plt.plot(list_max)
        # plt.plot(list_mean)
        # plt.plot(list_mean-3*list_std)
        # plt.plot(list_mean+3*list_std)
        # plt.legend('Max','Mean','Mean-3*STD','Mean+3*STD')
        i = 0
        while 1:

            im_list = []
            msk_list = []
            for ii in range(self.batch_size):
                im = np.load(list_flair[i+ii])
                # im = scipy.ndimage.interpolation.zoom(im,[2.,2.,2.],output=np.float32,order=0)
                #im = scipy.ndimage.interpolation.zoom(im, [0.5, 0.5, 0.5], output=np.float32, order=0)
                im_list.append(im)

                msk = np.load(list_gt[i+ii])
                # msk = np.maximum(0.,np.minimum(1.,10.*msk))
                # msk = scipy.ndimage.interpolation.zoom(msk,[2.,2.,2.],output=np.float32,order=0)
                #msk = scipy.ndimage.interpolation.zoom(msk, [0.5, 0.5, 0.5], output=np.float32, order=0)
                msk_list.append(msk)

                if i == num_images-self.batch_size:
                    i = 0
                else:
                    i = i + 1


            im = np.reshape(im_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            msk = np.reshape(msk_list,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
            yield im, msk
    def shuffle_data(self,f):
        idx = list(range(len(f[0])))
        shuffle(idx)
        f0 = []
        f1 = []
        for i in range(len(f[0])):
            f0.append(f[0][idx[i]])
            f1.append(f[1][idx[i]])
        f[0] = f0
        f[1] = f1
        return f

    def relu_plus(self,z):
        return tf.maximum(1.0,z)
    def PD_v2(self, u, reuse, training=False, batch_size= 1 ):
        self.batch_size_org = self.batch_size
        self.batch_size = batch_size
        with tf.variable_scope("Variational_Network", reuse=reuse):

            with tf.variable_scope("ResNet"):


                pre_delta, coded_unet = unet(self, inputs=u, kernel_size=3, padding='same', training=training)

                delta = tf.nn.relu(pre_delta) + 1.0

            with tf.variable_scope('Parameters'):

                conv3 = tf.layers.conv2d(coded_unet, filters=32, kernel_size=3, strides=2, padding='valid')
                conv3 = tf.layers.batch_normalization(conv3, training=training)
                conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=3, strides=2, padding='valid')
                conv4 = tf.layers.batch_normalization(conv4, training=training)
                conv5 = tf.layers.conv2d(conv4, filters=1, kernel_size=3, strides=2, padding='valid')
                conv5 = tf.layers.batch_normalization(conv5, training=training)

                fc = tf.reshape(conv5,shape=[self.batch_size,-1])
                fc = tf.layers.dense(fc,512,activation=tf.nn.relu)
                fc = tf.layers.batch_normalization(fc,training=training)
                fc = tf.layers.dense(fc,256,activation=tf.nn.relu)
                fc = tf.layers.batch_normalization(fc,training=training)
                fc = tf.layers.dense(fc,64,activation=tf.nn.relu)
                fc = tf.layers.batch_normalization(fc,training=training)
                par = tf.layers.dense(fc,2,activation=None)


            with tf.variable_scope("PD"):

                lda = 0.9*tf.nn.sigmoid(tf.reshape(par[:,0],[self.batch_size,1,1,1])) + 0.1
                alpha = 0.9*tf.nn.sigmoid(tf.reshape(par[:, 1], [self.batch_size, 1, 1, 1])) + 0.1

                f = u

                u, td, tp, sigma = PD(self, NitOut=4, NitIn=[40,20,10,5], u=u, delta=delta, alpha=alpha, lda=lda)

                self.batch_size = self.batch_size_org

                if self.pretrain:
                    u = tf.nn.relu(pre_delta)

            return u, td, tp, lda, alpha, delta, sigma, f

    def train(self):
        is_random = tf.placeholder(dtype=tf.bool,name='is_random')

        train_images = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        validation_images = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        # test_images = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        test_images = tf.placeholder(dtype=tf.float32,shape=[1, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])

        train_gt = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        validation_gt = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        # test_gt = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])
        test_gt = tf.placeholder(dtype=tf.float32,shape=[1, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])

        train_pack = tf.concat([train_images,train_gt],axis=-1)
        train_pack = tf.cond(is_random,lambda:tf.image.rot90(train_pack), lambda: train_pack)
        train_pack = tf.image.random_flip_up_down(train_pack)
        train_pack = tf.image.random_flip_left_right(train_pack)
        # train_pack = tf.image.random_crop(train_pack)
        train_images_da = tf.expand_dims(train_pack[:,:,:,0],-1)
        train_gt_da = tf.expand_dims(train_pack[:,:,:,1],-1)

        u_train, td, tp, lda, alpha, delta, sigma, f = self.PD_v2(train_images_da, reuse=False, training = True, batch_size=self.batch_size)
        u_validation, _, _, _, _, delta_v, _, _= self.PD_v2(validation_images, reuse=True, training= False, batch_size=self.batch_size)
        u_test, _, _, _, _, delta_t, _, _= self.PD_v2(test_images, reuse=True, training= False, batch_size= 1)


        with tf.variable_scope('Regularization'):
            TV2D = tf.reduce_mean(self.TV2D(delta))

        with tf.name_scope('Bayesian_Weights'):
            s1 = tf.Variable(tf.sqrt(0.5),name='s1')
            s2 = tf.Variable(0.1,name='s2')

            c1 = tf.div(1.,2.*(tf.square(s1)),name='c1')
            c2 = tf.div(1.,2.*(tf.square(s2)),name='c2')


        loss_l2 = tf.losses.mean_squared_error(labels=train_gt_da,predictions=u_train)
        U2 = tf.reduce_mean(tf.square(delta))
        # train_loss =  c1*tf.reduce_mean(1.-dice_coe(output=u_train,target=train_gt_da)) + s1 + 0.001*U2
        train_loss =  1.-dice_coe(output=u_train,target=train_gt_da)
        # train_loss = loss_l2

        validation_loss = 1.-dice_coe(output=u_validation,target=validation_gt)

        test_loss = 1.-dice_coe(output=u_test,target=test_gt)

        global_step = tf.Variable(0, trainable=False)

        # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        optimizer = optimizer_opt(self,loss=train_loss,var_list=var_list,global_step=global_step,show_gradients=self.show_gradients)


        saver = tf.train.Saver(var_list=var_list)
        # saver = tf.train.Saver()

        with tf.name_scope('All_Metrics'):
            dice_coe_t = dice_coe(output=u_train,target=train_gt_da)
            iou_coe_t = iou_coe(output=u_train,target=train_gt_da)

            dice_coe_v = dice_coe(output=u_validation,target=validation_gt)
            iou_coe_v = iou_coe(output=u_validation,target=validation_gt)

            dice_coe_test = dice_coe(output=u_test,target=test_gt)
            iou_coe_test = iou_coe(output=u_test,target=test_gt)
        
        # Summaries
        random_slice = tf.placeholder(dtype=tf.int32)

        with tf.name_scope('Training'):
            with tf.name_scope('Bayesian_weights'):
                tf.summary.scalar('c1',c1)
                # tf.summary.scalar('c2',c2)

            with tf.name_scope('Metrics'):
                tf.summary.scalar('Dice_coe',dice_coe_t)
                tf.summary.scalar('Iou_coe',iou_coe_t)

            with tf.name_scope('Images'):
                tf.summary.image('Input', train_images_da)

                out = u_train
                out = tf.cast(tf.reshape(out, (self.batch_size, self.IM_ROWS, self.IM_COLS, 1)), tf.float32)
                tf.summary.image('Output', out)

                sum_mask = train_gt_da
                sum_mask = tf.cast(tf.reshape(sum_mask, (self.batch_size, self.IM_ROWS, self.IM_COLS, 1)), tf.float32)
                tf.summary.image('GT', sum_mask)

                tf.summary.image('delta', delta)

            with tf.name_scope('Losses'):
                tf.summary.scalar('Loss',train_loss)
                # tf.summary.scalar('TV2D_Reg',TV2D)
                # tf.summary.scalar('U2_Reg',U2)

            with tf.name_scope('Parameters'):
                tf.summary.scalar('tp', tp)
                tf.summary.scalar('td', td)
                tf.summary.scalar('Lambda', lda[0,0,0,0])
                tf.summary.scalar('alpha', alpha[0,0,0,0])
                tf.summary.scalar('sigma', sigma)

        summary_train = tf.summary.merge_all()

        with tf.name_scope('Validation'):
            with tf.name_scope('Losses'):
                vl = tf.summary.scalar('Loss',validation_loss)
            with tf.name_scope('Metrics'):
                dv = tf.summary.scalar('Dice_coe',dice_coe_v)
                iv = tf.summary.scalar('Iou_coe',iou_coe_v)

            with tf.name_scope('Images'):
                inv = tf.summary.image('Input', validation_images)

                out = u_validation
                out = tf.cast(tf.reshape(out, (self.batch_size, self.IM_ROWS, self.IM_COLS, 1)), tf.float32)
                outv = tf.summary.image('Output', out)

                sum_mask = validation_gt
                sum_mask = tf.cast(tf.reshape(sum_mask, (self.batch_size, self.IM_ROWS, self.IM_COLS, 1)), tf.float32)
                gtv = tf.summary.image('GT', sum_mask)

                delv = tf.summary.image('delta', delta_v)

        summary_validation = tf.summary.merge(inputs=[vl, iv, dv, outv, gtv, inv, delv])

        with tf.name_scope('Test'):
            dice_test = tf.placeholder(dtype=tf.float32)
            iou_test = tf.placeholder(dtype=tf.float32)
            with tf.name_scope('Metrics'):
                d_test = tf.summary.scalar('Dice_coe',dice_test)
                i_test = tf.summary.scalar('Iou_coe',iou_test)


        summary_test = tf.summary.merge(inputs=[d_test, i_test])

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        with tf.Session() as sess:
            summary_writer_t = tf.summary.FileWriter(self.log_dir+'/train', sess.graph,filename_suffix='train')
            summary_writer_v = tf.summary.FileWriter(self.log_dir+'/validation', sess.graph,filename_suffix='valid')
            summary_writer_test = tf.summary.FileWriter(self.log_dir+'/test', sess.graph,filename_suffix='test')

            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.checkpoint)
                counter = np.load('counter.npy')
                epocas = np.load('epocas.npy')
            else:
                counter = 0
                epocas = 0

            t_f, v_f = self.create_list(data_path=self.data_dir)

            data_train = self.load_data_train(data_path=self.data_dir, list=t_f)
            data_valid = self.load_data_valid(data_path=self.data_dir, list=v_f)
            # data_test = self.load_data_test(data_path=self.data_dir, list=v_f)
            for i in tqdm(range(self.training_iters-counter)):

                if (counter % self.epoch_iteration)==0 or counter==0:
                    print('Shuffle Data')
                    t_f = self.shuffle_data(t_f)
                    v_f = self.shuffle_data(v_f)
                    data_train = self.load_data_train(data_path=self.data_dir, list=t_f)
                    data_valid = self.load_data_valid(data_path=self.data_dir, list=v_f)

                if not((counter % (self.epoch_iteration))==0):
                # if not((counter % (50))==0):
                    ti, tm = next(data_train)

                    _= sess.run([optimizer],feed_dict={train_images: ti, train_gt: tm, is_random: np.random.randint(0,2)})
                else:

                    ti, tm = next(data_train)
                    vi, vm = next(data_valid)

                    _, summary_str_t, summary_str_v, global_step_value = sess.run([optimizer, summary_train, summary_validation, global_step],feed_dict={train_images: ti, train_gt: tm,validation_images: vi, validation_gt: vm , is_random: np.random.randint(0,2)}, options = run_options, run_metadata = run_metadata)
                    summary_writer_t.add_run_metadata(run_metadata, 'step%d' % counter)
                    summary_writer_t.add_summary(summary_str_t, global_step_value)
                    summary_writer_t.flush()
                    summary_writer_v.add_run_metadata(run_metadata, 'step%d' % counter)
                    summary_writer_v.add_summary(summary_str_v, global_step_value)
                    summary_writer_v.flush()


                if (counter % self.epoch_iteration)==0 and not(counter==0):
                    epocas = epocas + 1
                    saver.save(sess, self.checkpoint)
                    np.save('counter.npy',counter)
                    np.save('epocas.npy',epocas)

                if self.is_test:
                    # print(counter)
                    if (counter % (self.test_each_epoch*self.epoch_iteration))==0 and not(counter==0):
                        dice_test_list = []
                        iou_test_list = []
                        test_list = np.load('./test_list.npy')
                        num_images = len(test_list[0])
                        list_flair = test_list[0]
                        list_gt = test_list[1]
                        for subject in tqdm(list_flair):
                            # vol_value = []
                            # vol_gt = []
                            # print ('un volumen')
                            for test_iter in range(155):
                                test_i = np.load(self.data_dir+'validation'+str(subject, 'utf-8') +str(test_iter)+'.npy')
                                test_i = np.reshape(test_i,newshape=[1,self.IM_ROWS,self.IM_COLS,1])
                                test_m = np.load(self.data_dir+'validation'+'/gt'+str(subject, 'utf-8')[3:]+str(test_iter)+'.npy')
                                test_m = np.reshape(test_m,newshape=[1,self.IM_ROWS,self.IM_COLS,1])

                                slice_value = sess.run([u_test],
                                    feed_dict={test_images: test_i, test_gt: test_m}, options=run_options, run_metadata=run_metadata)

                                if test_iter == 0:
                                    vol_value = slice_value
                                    vol_gt = test_m
                                else:
                                    vol_value = np.concatenate([vol_value,slice_value],axis=-1)
                                    vol_gt = np.concatenate([vol_gt,test_m],axis=-1)

                                # vol_value.append(slice_value[0][0,:,:,:])
                                # vol_gt.append(test_m[0,:,:,:])

                            vol_value = np.reshape(vol_value,[self.IM_ROWS,self.IM_COLS,155])
                            vol_gt = np.reshape(vol_gt,[self.IM_ROWS,self.IM_COLS,155])
                            dice_coe_test_value = self.np_dice(vol_value,vol_gt)
                            iou_coe_test_value = self.np_iou(vol_value,vol_gt)
                            dice_test_list.append(dice_coe_test_value)
                            iou_test_list.append(iou_coe_test_value)

                        plt.bar(np.arange(len(dice_test_list)), np.asarray(dice_test_list, dtype=np.float32))
                        plt.title('Test DICE Avg: ' + str(np.mean(dice_test_list)))
                        plt.savefig('test.png', bbox_inches='tight')

                        summary_str_test, global_step_value = sess.run([summary_test, global_step],feed_dict={dice_test: np.mean(dice_test_list), iou_test: np.mean(iou_test_list)})
                        summary_writer_test.add_run_metadata(run_metadata, 'step%d' % epocas)
                        summary_writer_test.add_summary(summary_str_test, global_step_value)
                        summary_writer_test.flush()

                counter += 1
            sess.close()

    def test(self):

        self.batch_size = 1
        self.restore = True
        test_images = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])

        test_gt = tf.placeholder(dtype=tf.float32,
                                 shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])

        u_test,td, tp, lda, alpha, delta, sigma, f = self.PD_v2(u=test_images, reuse=tf.AUTO_REUSE, training=False, batch_size=self.batch_size)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.checkpoint)

            dice_test_list = []
            iou_test_list = []
            test_list = np.load('./test_list.npy')
            num_images = len(test_list[0])
            list_flair = test_list[0]
            list_gt = test_list[1]
            for subject in tqdm(list_flair):

                for test_iter in range(155):
                    test_i = np.load(self.data_dir + 'validation' + str(subject, 'utf-8') + str(test_iter) + '.npy')
                    test_i = np.reshape(test_i, newshape=[1, self.IM_ROWS, self.IM_COLS, 1])
                    test_m = np.load(
                        self.data_dir + 'validation' + '/gt' + str(subject, 'utf-8')[3:] + str(test_iter) + '.npy')
                    test_m = np.reshape(test_m, newshape=[1, self.IM_ROWS, self.IM_COLS, 1])

                    slice_value = sess.run([u_test],
                                           feed_dict={test_images: test_i, test_gt: test_m})

                    if test_iter == 0:
                        vol_value = slice_value
                        vol_gt = test_m
                    else:
                        vol_value = np.concatenate([vol_value, slice_value], axis=-1)
                        vol_gt = np.concatenate([vol_gt, test_m], axis=-1)

                vol_value = np.reshape(vol_value, [self.IM_ROWS, self.IM_COLS, 155])
                vol_gt = np.reshape(vol_gt, [self.IM_ROWS, self.IM_COLS, 155])
                dice_coe_test_value = self.np_dice(vol_value, vol_gt)
                iou_coe_test_value = self.np_iou(vol_value, vol_gt)
                dice_test_list.append(dice_coe_test_value)
                iou_test_list.append(iou_coe_test_value)

            plt.bar(np.arange(len(dice_test_list)), np.asarray(dice_test_list, dtype=np.float32))
            plt.title('Test DICE Avg: '+str(np.mean(dice_test_list)))
            plt.savefig('test.png', bbox_inches='tight')

            print(np.mean(dice_test_list))