#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import caffe
import argparse
from caffe.coord_map import crop
import numpy as np

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    """
    Def conv layer with ReLU. bottom: input blob; nout: output channel.
    """
    conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, 
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, caffe.layers.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2, pad=0):
    """
    Def max pooling layer.
    """
    pool = caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)
    return pool

def dil_conv_relu(bottom, nout, ks=3, stride=1, pad=1, dil=1):
    """
    Def dilated conv layer with ReLU. bottom: input blob; nout: output channel.
    """
    conv = caffe.layers.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, dilation=dil,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, caffe.layers.ReLU(conv, in_place=True)

def write_net(dataset):
    """
    Generate train.prototxt and test.prototxt.
    Dataset: 'sklarge' for SK-LARGE; 'sk506' for SK506; 'whsymmax' for wh-SYMMAX; 'sympascal' for SYM-PASCAL; 'symmax300' for SYMMAX300.
    """
    net = caffe.NetSpec()
    datalayer_params = dict(data_dir='/home/wangyukang/dataset/', dataset=dataset, seed=123, mean=(103.939, 116.779, 123.68))
    net.image, net.flux, net.dilmask = caffe.layers.Python(module='pylayerUtils', layer='DataLayer', ntop=3, param_str=str(datalayer_params))

    net.conv1_1, net.relu1_1 = conv_relu(net.image, 64, pad=35)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 64)
    net.pool1 = max_pool(net.relu1_2)

    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 128)
    net.pool2 = max_pool(net.relu2_2)

    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 256)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 256)
    net.pool3 = max_pool(net.relu3_3)

    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 512)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 512)
    net.pool4 = max_pool(net.relu4_3)

    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 512)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 512)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 512)

    net.d2conv, net.d2relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=2, dil=2)
    net.d4conv, net.d4relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=4, dil=4)
    net.d8conv, net.d8relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=8, dil=8)
    net.d16conv, net.d16relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=16, dil=16)

    bottom_layers = [net.d2relu, net.d4relu, net.d8relu, net.d16relu]
    net.dconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.sdconv = caffe.layers.Convolution(net.dconcat, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.sdrelu = caffe.layers.ReLU(net.sdconv, in_place=True)
    net.upsd = caffe.layers.Deconvolution(net.sdrelu,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.sdcrop = crop(net.upsd, net.relu3_3)

    net.sconv5 = caffe.layers.Convolution(net.relu5_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu5 = caffe.layers.ReLU(net.sconv5, in_place=True)
    net.ups5 = caffe.layers.Deconvolution(net.srelu5,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.scrop5 = crop(net.ups5, net.relu3_3)

    net.sconv4 = caffe.layers.Convolution(net.relu4_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu4 = caffe.layers.ReLU(net.sconv4, in_place=True)
    net.ups4 = caffe.layers.Deconvolution(net.srelu4,convolution_param=dict(num_output=256, kernel_size=4, stride=2, pad=1, bias_term=False), param=[dict(lr_mult=0)])
    net.scrop4 = crop(net.ups4, net.relu3_3)

    net.sconv3 = caffe.layers.Convolution(net.relu3_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.0001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu3 = caffe.layers.ReLU(net.sconv3, in_place=True)

    bottom_layers = [net.srelu3, net.scrop4, net.scrop5, net.sdcrop]
    net.sconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.fconv1, net.frelu1 = conv_relu(net.sconcat, 512, ks=1, pad=0)
    net.fconv2, net.frelu2 = conv_relu(net.frelu1, 512, ks=1, pad=0)
    net.fconv3 = caffe.layers.Convolution(net.frelu2, kernel_size=1, stride=1, num_output=2, pad=0,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.fup = caffe.layers.Deconvolution(net.fconv3,convolution_param=dict(num_output=2, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.fcrop = crop(net.fup, net.image)
    net.loss = caffe.layers.Python(net.fcrop, net.flux, net.dilmask, module='pylayerUtils', layer='WeightedEuclideanLossLayer', loss_weight=1)

    with open('train.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
    with open('test.prototxt', 'w') as f:
        f.write(str(net.to_proto()))

def write_deploy():
    """
    Generate deploy.prototxt.
    """
    net = caffe.NetSpec()
    net.data = caffe.layers.Input(input_param={'shape':{'dim':[1,3,512,512]}})

    net.conv1_1, net.relu1_1 = conv_relu(net.data, 64, pad=35)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 64)
    net.pool1 = max_pool(net.relu1_2)

    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 128)
    net.pool2 = max_pool(net.relu2_2)

    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 256)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 256)
    net.pool3 = max_pool(net.relu3_3)

    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 512)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 512)
    net.pool4 = max_pool(net.relu4_3)

    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 512)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 512)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 512)

    net.d2conv, net.d2relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=2, dil=2)
    net.d4conv, net.d4relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=4, dil=4)
    net.d8conv, net.d8relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=8, dil=8)
    net.d16conv, net.d16relu = dil_conv_relu(net.relu5_3, 128, ks=3, stride=1, pad=16, dil=16)

    bottom_layers = [net.d2relu, net.d4relu, net.d8relu, net.d16relu]
    net.dconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.sdconv = caffe.layers.Convolution(net.dconcat, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.sdrelu = caffe.layers.ReLU(net.sdconv, in_place=True)
    net.upsd = caffe.layers.Deconvolution(net.sdrelu,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.sdcrop = crop(net.upsd, net.relu3_3)

    net.sconv5 = caffe.layers.Convolution(net.relu5_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.01},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu5 = caffe.layers.ReLU(net.sconv5, in_place=True)
    net.ups5 = caffe.layers.Deconvolution(net.srelu5,convolution_param=dict(num_output=256, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.scrop5 = crop(net.ups5, net.relu3_3)

    net.sconv4 = caffe.layers.Convolution(net.relu4_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu4 = caffe.layers.ReLU(net.sconv4, in_place=True)
    net.ups4 = caffe.layers.Deconvolution(net.srelu4,convolution_param=dict(num_output=256, kernel_size=4, stride=2, pad=1, bias_term=False), param=[dict(lr_mult=0)])
    net.scrop4 = crop(net.ups4, net.relu3_3)

    net.sconv3 = caffe.layers.Convolution(net.relu3_3, kernel_size=1, stride=1, num_output=256, pad=0,
        weight_filler={"type": "gaussian", "std": 0.0001},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.srelu3 = caffe.layers.ReLU(net.sconv3, in_place=True)

    bottom_layers = [net.srelu3, net.scrop4, net.scrop5, net.sdcrop]
    net.sconcat = caffe.layers.Concat(*bottom_layers, concat_param=dict(concat_dim=1))

    net.fconv1, net.frelu1 = conv_relu(net.sconcat, 512, ks=1, pad=0)
    net.fconv2, net.frelu2 = conv_relu(net.frelu1, 512, ks=1, pad=0)
    net.fconv3 = caffe.layers.Convolution(net.frelu2, kernel_size=1, stride=1, num_output=2, pad=0,
        weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    net.fup = caffe.layers.Deconvolution(net.fconv3,convolution_param=dict(num_output=2, kernel_size=8, stride=4, pad=2, bias_term=False), param=[dict(lr_mult=0)])
    net.fcrop = crop(net.fup, net.data)

    with open('deploy.prototxt', 'w') as f:
        f.write(str(net.to_proto()))

def write_solver(base_lr, iters, snapshot):
    """
    Generate solver.prototxt.
    base_lr: learning rate;
    iters: max iterations;
    snapshot: the prefix of saved models.
    """
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()
    sovler_string.train_net = 'train.prototxt'
    sovler_string.test_net.append('test.prototxt')
    sovler_string.test_iter.append(500)
    sovler_string.test_interval = 999999
    sovler_string.type = 'Adam'
    sovler_string.base_lr = base_lr
    sovler_string.momentum = 0.9
    sovler_string.momentum2 = 0.999
    sovler_string.lr_policy = 'fixed'
    sovler_string.display = 100
    sovler_string.average_loss = 100
    sovler_string.max_iter = iters
    sovler_string.snapshot = 10000
    sovler_string.snapshot_prefix = snapshot
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
    sovler_string.test_initialization = 0

    with open('solver.prototxt', 'w') as f:
        f.write(str(sovler_string))

def train(initmodel, gpu):
    """
    Train the net.
    """
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    solver = caffe.AdamSolver('solver.prototxt')
    if initmodel:
        solver.net.copy_from(initmodel)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp(solver.net, interp_layers)
    solver.step(solver.param.max_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset name.")
    parser.add_argument("--initmodel", help="Init caffemodel.")
    parser.add_argument("--gpu", required=True, type=int, help="Device ids.")
    args = parser.parse_args()

    if not os.path.isdir('snapshot_1e-4'):
        os.makedirs('snapshot_1e-4')
    if not os.path.isdir('snapshot_1e-5'):
        os.makedirs('snapshot_1e-5')
    if not os.path.isdir('test'):
        os.makedirs('test')

    write_net(args.dataset)
    write_solver(1e-4, 80000, 'snapshot_1e-4/'+args.dataset)
    train(args.initmodel, args.gpu)

    write_solver(1e-5, 40000, 'snapshot_1e-5/'+args.dataset)
    train('snapshot_1e-4/'+args.dataset+'_iter_80000.caffemodel', args.gpu)
    write_deploy()
