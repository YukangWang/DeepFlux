import os
import numpy as np
import caffe
import cv2
import sys
import math

def trans_n8(mask, px, py):
    M = np.array([[1,0,py],[0,1,px]]).astype(np.float32)
    trans = cv2.warpAffine(mask, M, mask.shape[::-1])
    return trans

def flux_to_skl(flux, threshold, dks, eks):
    magnitude, angle = cv2.cartToPolar(flux[0][0], flux[0][1])
    vmax = float(magnitude.max())
    mask = (magnitude > threshold).astype(np.uint8)
    mask_rev = (mask == 0).astype(np.uint8)
    ending = trans_n8(mask_rev, -1, -1)*mask*np.logical_and(angle >= math.pi/8, angle < 3*math.pi/8) \
           + trans_n8(mask_rev, 0, -1)*mask*np.logical_and(angle >= 3*math.pi/8, angle < 5*math.pi/8) \
           + trans_n8(mask_rev, 1, -1)*mask*np.logical_and(angle >= 5*math.pi/8, angle < 7*math.pi/8) \
           + trans_n8(mask_rev, 1, 0)*mask*np.logical_and(angle >= 7*math.pi/8, angle < 9*math.pi/8) \
           + trans_n8(mask_rev, 1, 1)*mask*np.logical_and(angle >= 9*math.pi/8, angle < 11*math.pi/8) \
           + trans_n8(mask_rev, 0, 1)*mask*np.logical_and(angle >= 11*math.pi/8, angle < 13*math.pi/8) \
           + trans_n8(mask_rev, -1, 1)*mask*np.logical_and(angle >= 13*math.pi/8, angle < 15*math.pi/8) \
           + trans_n8(mask_rev, -1, 0)*mask*np.logical_or(angle < math.pi/8, angle >= 15*math.pi/8)
    ending = (ending > 0).astype(np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
    ending = cv2.dilate(ending, element)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eks, eks))
    ending = cv2.erode(ending, element)
    skl = ending * (vmax - magnitude) / vmax
    return skl

proto = sys.argv[1]
model = sys.argv[2]
gpu = int(sys.argv[3])
input_dir = sys.argv[4]
threshold = float(sys.argv[5])
dks = int(sys.argv[6])
eks = int(sys.argv[7])
output_dir = sys.argv[8]

caffe.set_device(int(gpu))
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

net = caffe.Net(proto, model, caffe.TEST)
files = os.listdir(input_dir)
for num in range(len(files)):
    image = cv2.imread(input_dir+files[num],1)
    image = image.astype(np.float32)
    image -= np.array((103.939, 116.779, 123.68))
    in_ = image.transpose((2,0,1))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    flux = net.blobs['fcrop'].data
    skl = flux_to_skl(flux, threshold, dks, eks)
    cv2.imwrite(output_dir+'/'+files[num][:-4]+'.png', 255*skl)
    print num
