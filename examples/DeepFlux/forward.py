import os
import numpy as np
from PIL import Image
from scipy.misc import imresize
import caffe
import cv2
import scipy.io as sio
import sys
import getopt

opts,args = getopt.getopt(sys.argv[1:],'-i:-m:-d:-o:',['gpu=','model=','dir=','out='])
for opt_name, opt_value in opts:
    if opt_name in ('-i','--gpu'):
        device_id = opt_value
    if opt_name in ('-m','--model'):
        model_weights = opt_value
    if opt_name in ('-d','--dir'):
        file_dir = opt_value
    if opt_name in ('-o','--out'):
        save_dir = opt_value

caffe.set_device(int(device_id))
caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
net = caffe.Net(model_def, model_weights, caffe.TEST)

files = os.listdir(file_dir)
for num in range(len(files)):
    image = cv2.imread(file_dir+files[num],1)
    image = image.astype(np.float64)
    image -= np.array((103.939, 116.779, 123.68))
    in_ = image.transpose((2,0,1))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['fcrop'].data[0]
    sio.savemat(save_dir+files[num][:-4]+'.mat',{'GTcls':out})
    print num
