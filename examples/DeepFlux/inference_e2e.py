import os
import numpy as np
import caffe
import cv2
import sys
import math

proto = sys.argv[1]
model = sys.argv[2]
gpu = int(sys.argv[3])
input_dir = sys.argv[4]
output_dir = sys.argv[5]

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
    skl = net.blobs['e2ecrop'].data[0][0]
    cv2.imwrite(output_dir+'/'+files[num][:-4]+'.png', 255*skl)
    print num
