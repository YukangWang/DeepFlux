import caffe
import numpy as np
import cv2
import random

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # data layer config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.dataset = params['dataset']
        self.seed = params['seed']
        self.mean = np.array(params['mean'])
        self.random = True

        # three tops: image, flux and dilmask
        if len(top) != 3:
            raise Exception("Need to define three tops: image, flux and dilmask.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # read filename list for each dataset here
        if self.dataset == 'sklarge':
            self.fnLst = open(self.data_dir+'SK-LARGE/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'sk506':
            self.fnLst = open(self.data_dir+'SK506/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'whsymmax':
            self.fnLst = open(self.data_dir+'wh-symmax/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'sympascal':
            self.fnLst = open(self.data_dir+'SymPASCAL-by-KZ/aug_data/train_pair.lst').readlines()
        elif self.dataset == 'symmax300':
            self.fnLst = open(self.data_dir+'SYMMAX300/aug_data/train_pair.lst').readlines()
        else:
            raise Exception("Invalid dataset.")

        # randomization: seed and pick
        self.idx = 0
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.fnLst)-1)

    def reshape(self, bottom, top):
        # load image, flux and dilmask
        if self.dataset == 'sklarge':
            self.image, self.flux, self.dilmask = self.loadsklarge(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'sk506':
            self.image, self.flux, self.dilmask = self.loadsk506(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'whsymmax':
            self.image, self.flux, self.dilmask = self.loadwhsymmax(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'sympascal':
            self.image, self.flux, self.dilmask = self.loadsympascal(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        elif self.dataset == 'symmax300':
            self.image, self.flux, self.dilmask = self.loadsymmax300(self.fnLst[self.idx].split()[0],self.fnLst[self.idx].split()[1])
        else:
            raise Exception("Invalid dataset.")

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.image.shape)
        top[1].reshape(1, *self.flux.shape)
        top[2].reshape(1, *self.dilmask.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.image
        top[1].data[...] = self.flux
        top[2].data[...] = self.dilmask

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
        # load image and skeleton
        image = cv2.imread('{}/SK-LARGE/{}'.format(self.data_dir, imgidx),1)
        skeleton = cv2.imread('{}/SK-LARGE/{}'.format(self.data_dir, gtidx),0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute flux and dilmask
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        skl = (skl>0).astype(np.float32)
        skl = skl[np.newaxis, ...]

        return image, flux, skl

    def loadsk506(self, imgidx, gtidx):
        # load image and skeleton
        image = cv2.imread('{}/SK506/{}'.format(self.data_dir, imgidx),1)
        skeleton = cv2.imread('{}/SK506/{}'.format(self.data_dir, gtidx),0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute flux and dilmask
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        skl = (skl>0).astype(np.float32)
        skl = skl[np.newaxis, ...]

        return image, flux, skl

    def loadwhsymmax(self, imgidx, gtidx):
        # load image and skeleton
        image = cv2.imread('{}/wh-symmax/{}'.format(self.data_dir, imgidx),1)
        skeleton = cv2.imread('{}/wh-symmax/{}'.format(self.data_dir, gtidx),0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute flux and dilmask
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        skl = (skl>0).astype(np.float32)
        skl = skl[np.newaxis, ...]

        return image, flux, skl

    def loadsympascal(self, imgidx, gtidx):
        # load image and skeleton
        image = cv2.imread('{}/SymPASCAL-by-KZ/{}'.format(self.data_dir, imgidx),1)
        skeleton = cv2.imread('{}/SymPASCAL-by-KZ/{}'.format(self.data_dir, gtidx),0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute flux and dilmask
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        skl = (skl>0).astype(np.float32)
        skl = skl[np.newaxis, ...]

        return image, flux, skl

    def loadsymmax300(self, imgidx, gtidx):
        # load image and skeleton
        image = cv2.imread('{}/SYMMAX300/{}'.format(self.data_dir, imgidx),1)
        skeleton = cv2.imread('{}/SYMMAX300/{}'.format(self.data_dir, gtidx),0)
        skeleton = (skeleton > 0).astype(np.uint8)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute flux and dilmask
        kernel = np.ones((15,15), np.uint8)
        dilmask = cv2.dilate(skeleton, kernel)
        rev = 1-skeleton
        height = rev.shape[0]
        width = rev.shape[1]
        rev = (rev > 0).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[rev > 0] = 0
        place = np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height, width))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(rev.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel

        dist = np.sqrt(np.sum(diff**2, axis = 0))

        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
        direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

        direction[0] = direction[0]*(dilmask > 0)
        direction[1] = direction[1]*(dilmask > 0)

        flux = -1*np.stack((direction[0], direction[1]))

        skl = (skl>0).astype(np.float32)
        skl = skl[np.newaxis, ...]

        return image, flux, skl

class WeightedEuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check inputs
        if len(bottom) != 2:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # def dist and weight for backpropagation
        self.distL1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.distL2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightPos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightNeg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # L1 and L2 distance
        self.distL1 = bottom[0].data - bottom[1].data
        self.distL2 = self.distL1**2
        # the amount of positive and negative pixels
        mask = np.logical_or((bottom[1].data[0][0]!=0), (bottom[1].data[0][1]!=0)).astype(np.float32)
        regionPos = (mask>0)
        regionNeg = (mask==0)
        sumPos = np.sum(regionPos)
        sumNeg = np.sum(regionNeg)
        # balanced weight for positive and negative pixels
        self.weightPos[0][0] = sumNeg/float(sumPos+sumNeg)*regionPos
        self.weightPos[0][1] = sumNeg/float(sumPos+sumNeg)*regionPos
        self.weightNeg[0][0] = sumPos/float(sumPos+sumNeg)*regionNeg
        self.weightNeg[0][1] = sumPos/float(sumPos+sumNeg)*regionNeg
        # total loss
        top[0].data[...] = np.sum(self.distL2*(self.weightPos + self.weightNeg)) / bottom[0].num / 2. / np.sum(self.weightPos + self.weightNeg)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.distL1*(self.weightPos + self.weightNeg) / bottom[0].num
        bottom[1].diff[...] = 0

class WeightedSigmoidCrossEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # define diff and weight maps for backpropagation
        self.sce = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightPos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightNeg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # compute loss and diff
        self.sce = -1*(bottom[0].data*(bottom[1].data-(bottom[0].data>=0))-np.log2(1+np.exp(bottom[0].data-2*bottom[0].data*(bottom[0].data>=0))))
        self.diff = 1.0/(1+np.exp(-1*bottom[0].data))-bottom[1].data
        # the amount of positive and negative pixels
        regionPos = (bottom[1].data>0)
        regionNeg = (bottom[1].data==0)
        sumPos = np.sum(regionPos)
        sumNeg = np.sum(regionNeg)
        # balanced weight for positive and negative pixels
        self.weightPos = sumNeg/float(sumPos+sumNeg)*regionPos
        self.weightNeg = sumPos/float(sumPos+sumNeg)*regionNeg
        # total loss
        top[0].data[...] = np.sum(self.sce*(self.weightPos + self.weightNeg)) / bottom[0].num / np.sum(self.weightPos + self.weightNeg)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.diff*(self.weightPos + self.weightNeg) / bottom[0].num
        bottom[1].diff[...] = 0