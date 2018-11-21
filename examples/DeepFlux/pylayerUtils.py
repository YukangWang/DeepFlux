import caffe

import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy import sparse

import random
import os
import cv2
import time
import math
import scipy.io as sio

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # data layer config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.dataset = params['dataset']

        # three tops: data, label and weight
        if len(top) != 3:
            raise Exception("Need to define three tops: data, label and weight.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # set image root for each dataset here
        if self.dataset == 'sklarge':
            self.fnLst = open(self.data_dir+'symmetry_detection/SK-LARGE/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'sk506':
            self.fnLst = open(self.data_dir+'symmetry_detection/SK506/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'whsymmax':
            self.fnLst = open(self.data_dir+'symmetry_detection/wh-symmax/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'sympascal':
            self.fnLst = open(self.data_dir+'symmetry_detection/SymPASCAL-by-KZ/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'symmax300':
            self.fnLst = open(self.data_dir+'symmetry_detection/SYMMAX300/aug_data/train_pair.lst').readlines()
        else:
            raise Exception("Invalid dataset.")
        self.idx = 0

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.fnLst)-1)

    def reshape(self, bottom, top):
        # load image, label and weight
        if self.dataset == 'sklarge':
            self.data, self.label, self.weight = self.loadsklarge(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'sk506':
            self.data, self.label, self.weight = self.loadsk506(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'whsymmax':
            self.data, self.label, self.weight = self.loadwhsymmax(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'sympascal':
            self.data, self.label, self.weight = self.loadsympascal(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'symmax300':
            self.data, self.label, self.weight = self.loadsymmax300(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        else:
            raise Exception("Invalid dataset.")

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.fnLst)-1)
        else:
            self.idx += 1
            if self.idx == len(self.fnLst):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def loadsklarge(self, imgidx, gtidx):
        # load img and skl
        image = cv2.imread('{}/symmetry_detection/SK-LARGE/{}'.format(self.data_dir, imgidx),1)
        inputImage = image.astype(np.float32)
        inputImage -= self.mean
        inputImage = inputImage.transpose((2,0,1))
        skl = cv2.imread('{}/symmetry_detection/SK-LARGE/{}'.format(self.data_dir, gtidx),0)
        skl = 255*(skl > 0).astype(np.uint8)

        # compute ground truth
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.dilate(skl, kernel)
        img = 255-skl
        height = img.shape[0]
        width = img.shape[1]
        img = img > 128
        img = img.astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((3, height, width), dtype=np.float32)
        direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
        direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
        direction[2,img > 0] = 1

        direction[0] = direction[0]*(mask > 128)
        direction[1] = direction[1]*(mask > 128)
        direction[2] = direction[2]*(mask > 128)

        # compute weight map
        inputGtxy = -1*np.stack((direction[0], direction[1]))
        posRegion = direction[2]>0
        weights = 1*posRegion
        inputWeight = weights[np.newaxis, ...]

        return inputImage, inputGtxy, inputWeight

    def loadsk506(self, imgidx, gtidx):
        # load img and skl
        image = cv2.imread('{}/symmetry_detection/SK506/{}'.format(self.data_dir, imgidx),1)
        inputImage = image.astype(np.float32)
        inputImage -= self.mean
        inputImage = inputImage.transpose((2,0,1))
        skl = cv2.imread('{}/symmetry_detection/SK506/{}'.format(self.data_dir, gtidx),0)
        skl = 255*(skl > 0).astype(np.uint8)

        # compute ground truth
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.dilate(skl, kernel)
        img = 255-skl
        height = img.shape[0]
        width = img.shape[1]
        img = img > 128
        img = img.astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((3, height, width), dtype=np.float32)
        direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
        direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
        direction[2,img > 0] = 1

        direction[0] = direction[0]*(mask > 128)
        direction[1] = direction[1]*(mask > 128)
        direction[2] = direction[2]*(mask > 128)

        # compute weight map
        inputGtxy = -1*np.stack((direction[0], direction[1]))
        posRegion = direction[2]>0
        weights = 1*posRegion
        inputWeight = weights[np.newaxis, ...]

        return inputImage, inputGtxy, inputWeight

    def loadwhsymmax(self, imgidx, gtidx):
        # load img and skl
        image = cv2.imread('{}/symmetry_detection/wh-symmax/{}'.format(self.data_dir, imgidx),1)
        inputImage = image.astype(np.float32)
        inputImage -= self.mean
        inputImage = inputImage.transpose((2,0,1))
        skl = cv2.imread('{}/symmetry_detection/wh-symmax/{}'.format(self.data_dir, gtidx),0)
        skl = 255*(skl > 0).astype(np.uint8)

        # compute ground truth
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.dilate(skl, kernel)
        img = 255-skl
        height = img.shape[0]
        width = img.shape[1]
        img = img > 128
        img = img.astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((3, height, width), dtype=np.float32)
        direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
        direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
        direction[2,img > 0] = 1

        direction[0] = direction[0]*(mask > 128)
        direction[1] = direction[1]*(mask > 128)
        direction[2] = direction[2]*(mask > 128)

        # compute weight map
        inputGtxy = -1*np.stack((direction[0], direction[1]))
        posRegion = direction[2]>0
        weights = 1*posRegion
        inputWeight = weights[np.newaxis, ...]

        return inputImage, inputGtxy, inputWeight

    def loadsympascal(self, imgidx, gtidx):
        # load img and skl
        image = cv2.imread('{}/symmetry_detection/SymPASCAL-by-KZ/{}'.format(self.data_dir, imgidx),1)
        inputImage = image.astype(np.float32)
        inputImage -= self.mean
        inputImage = inputImage.transpose((2,0,1))
        skl = cv2.imread('{}/symmetry_detection/SymPASCAL-by-KZ/{}'.format(self.data_dir, gtidx),0)
        skl = 255*(skl > 0).astype(np.uint8)

        # compute ground truth
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.dilate(skl, kernel)
        img = 255-skl
        height = img.shape[0]
        width = img.shape[1]
        img = img > 128
        img = img.astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((3, height, width), dtype=np.float32)
        direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
        direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
        direction[2,img > 0] = 1

        direction[0] = direction[0]*(mask > 128)
        direction[1] = direction[1]*(mask > 128)
        direction[2] = direction[2]*(mask > 128)

        # compute weight map
        inputGtxy = -1*np.stack((direction[0], direction[1]))
        posRegion = direction[2]>0
        weights = 1*posRegion
        inputWeight = weights[np.newaxis, ...]

        return inputImage, inputGtxy, inputWeight

    def loadsymmax300(self, imgidx, gtidx):
        # load img and skl
        image = cv2.imread('{}/symmetry_detection/SYMMAX300/{}'.format(self.data_dir, imgidx),1)
        inputImage = image.astype(np.float32)
        inputImage -= self.mean
        inputImage = inputImage.transpose((2,0,1))
        skl = cv2.imread('{}/symmetry_detection/SYMMAX300/{}'.format(self.data_dir, gtidx),0)
        skl = 255*(skl > 0).astype(np.uint8)

        # compute ground truth
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.dilate(skl, kernel)
        img = 255-skl
        height = img.shape[0]
        width = img.shape[1]
        img = img > 128
        img = img.astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((3, height, width), dtype=np.float32)
        direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
        direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
        direction[2,img > 0] = 1

        direction[0] = direction[0]*(mask > 128)
        direction[1] = direction[1]*(mask > 128)
        direction[2] = direction[2]*(mask > 128)

        # compute weight map
        inputGtxy = -1*np.stack((direction[0], direction[1]))
        posRegion = direction[2]>0
        weights = 1*posRegion
        inputWeight = weights[np.newaxis, ...]

        return inputImage, inputGtxy, inputWeight

class WeightedEuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # define diff and weight maps for backpropagation
        self.distL1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.distL2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightPos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightNeg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # compute l1 and l2 distance
        self.distL1 = bottom[0].data - bottom[1].data
        self.distL2 = self.distL1**2
        # pos sum and neg sum
        posRegion = (bottom[2].data>0)
        negRegion = (bottom[2].data==0)
        posSum = np.sum(posRegion)
        negSum = np.sum(negRegion)
        # weight for posPixel and negPixel
        self.weightPos[0][0] = float(negSum)/float(posSum+negSum)*posRegion
        self.weightPos[0][1] = float(negSum)/float(posSum+negSum)*posRegion
        self.weightNeg[0][0] = float(posSum)/float(posSum+negSum)*negRegion
        self.weightNeg[0][1] = float(posSum)/float(posSum+negSum)*negRegion
        # total loss
        top[0].data[...] = np.sum((self.distL1**2)*(self.weightPos + self.weightNeg)) / bottom[0].num / 2. / np.sum(self.weightPos + self.weightNeg)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.distL1*(self.weightPos + self.weightNeg) / bottom[0].num
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0
