import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
import platform
import subprocess as sp
import numpy as np
import tensorflow as tf
from util import *
from transform import grad, curl, advect, rotate, rot_mat
import sys
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='data/smoke_gun')
parser.add_argument("--log_dir", type=str, default='log/smoke_gun')
parser.add_argument("--npz2vdb_dir", type=str, default='data\\npz2vdb')
parser.add_argument("--tag", type=str, default='test')
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--model_path", type=str, default='data/model/tensorflow_inception_graph.pb')
parser.add_argument("--pool1", type=str2bool, default=True)

parser.add_argument("--transmit", type=float, default=0.1)
parser.add_argument("--rotate", type=str2bool, default=True)
parser.add_argument('--phi0', type=int, default=-5) # latitude (elevation) start
parser.add_argument('--phi1', type=int, default=5) # latitude end
parser.add_argument('--phi_unit', type=int, default=5)
parser.add_argument('--theta0', type=int, default=-10) # longitude start
parser.add_argument('--theta1', type=int, default=10) # longitude end
parser.add_argument('--theta_unit', type=int, default=10)
parser.add_argument('--v_batch', type=int, default=1, help='# of rotation matrix for batch process')
parser.add_argument('--n_views', type=int, default=9, help='# of view points')
parser.add_argument('--sample_type', type=str, default='poisson',
                    choices=['uniform', 'poisson', 'both'])

parser.add_argument("--target_frame", type=int, default=70)
parser.add_argument("--num_frames", type=int, default=1)
parser.add_argument("--window_size", type=int, default=1)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--mask", type=str2bool, default=True)

parser.add_argument("--field_type", type=str, default='field',
                    choices=['field', 'velocity', 'density'])
parser.add_argument("--w_field", type=float, default=1, help='weight between pot. and str.')
parser.add_argument("--adv_order", type=int, default=2, choices=[1,2], help='SL or MACCORMACK')
parser.add_argument("--resize_scale", type=float, default=1.0)

parser.add_argument("--content_layer", type=str, default='mixed4d_3x3_bottleneck_pre_relu')
parser.add_argument("--content_channel", type=int, default=139)
parser.add_argument("--style_layer", type=str, default='conv2d2,mixed3b,mixed4b')
parser.add_argument("--w_content", type=float, default=1)
parser.add_argument("--w_content_amp", type=float, default=100)
parser.add_argument("--w_style", type=float, default=0)
parser.add_argument("--w_style_layer", type=str, default='1,1,1')
parser.add_argument("--content_target", type=str, default='')
parser.add_argument("--style_target", type=str, default='')
parser.add_argument("--top_k", type=int, default=5)

parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--lap_n", type=int, default=3)
parser.add_argument("--octave_n", type=int, default=3)
parser.add_argument("--octave_scale", type=float, default=1.8)
parser.add_argument("--g_sigma", type=float, default=1.2)

#### HOUDINI
parser.add_argument("--houdini", type=str2bool, default=False)
parser.add_argument("--single_frame", type=str2bool, default=False)
parser.add_argument("--iter_seg", type=int, default=0)
parser.add_argument("--style_path", type=str, default='')
#### HOUDINI


class Styler(object):
    def __init__(self, self_dict):
        # get arguments
        for arg in self_dict: setattr(self, arg, self_dict[arg])
        self.rng = np.random.RandomState(self.seed)
        tf.set_random_seed(self.seed)

        # network setting
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        if self.houdini:
            self.model_path = self.style_path + "/" + self.model_path
        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # fix checkerboard artifacts: ksize should be divisible by the stride size
        # but it changes scale
        if self.pool1:
            for n in graph_def.node:
                if 'conv2d0_pre_relu/conv' in n.name:
                    n.attr['strides'].list.i[1:3] = [1,1]


        # density input
        # shape: [D,H,W]
        d_shp = [None,None,None]
        self.d = tf.placeholder(dtype=tf.float32, shape=d_shp, name='density')

        # add batch dim / channel dim
        # shape: [1,D,H,W,1]
        d = tf.expand_dims(tf.expand_dims(self.d, axis=0), axis=-1)
        
        ######
        # sequence stylization
        self.d_opt = tf.placeholder(dtype=tf.float32, name='opt')

        if 'field' in self.field_type:
            if self.w_field == 1:
                self.c = 1
            elif self.w_field == 0:
                self.c = 3
            else:
                self.c = 4 # scalar (1) + vector field (3)
        elif 'density' in self.field_type:
            self.c = 1 # scalar field
        else:
            self.c = 3 # vector field

        if 'field' in self.field_type:
            d_opt = self.d_opt[:,:,::-1] * tf.to_float(tf.shape(self.d_opt)[2])
            if self.w_field == 1:
                self.v_ = grad(d_opt)
            elif self.w_field == 0:
                self.v_ = curl(d_opt)
            else:
                pot = d_opt[...,0,None]
                strf = d_opt[...,1:]
                self.v_p = grad(pot)
                self.v_s = curl(strf)
                self.v_ = self.v_p*self.w_field + self.v_s*(1-self.w_field)

            v = self.v_[:,:,::-1]
            vx = v[...,0] / tf.to_float(tf.shape(v)[3]) 
            vy = -v[...,1] / tf.to_float(tf.shape(v)[2])
            vz = v[...,2] / tf.to_float(tf.shape(v)[1])
            v = tf.stack([vz,vy,vx], axis=-1)
            d = advect(d, v, order=self.adv_order, is_3d=True)
        elif 'velocity' in self.field_type:
            v = self.d_opt # [1,D,H,W,3]
            d = advect(d, v, order=self.adv_order, is_3d=True)
        else:
            # stylize by addition
            d += self.d_opt # [1,D,H,W,1]

        self.b_num = self.v_batch
        ######

        ######
        # velocity fields to advect gradients [B,D,H,W,3]
        if self.window_size > 1:
            self.v = tf.placeholder(dtype=tf.float32, name='velocity')
            self.g = tf.placeholder(dtype=tf.float32, name='gradient')
            self.adv = advect(self.g, self.v, order=self.adv_order, is_3d=True)
        ######

        # value clipping (d >= 0)
        d = tf.maximum(d, 0)

        # stylized 3d result
        self.d_out = d

        if self.rotate:
            d, self.rot_mat = rotate(d) # [b,D,H,W,1]

            # compute rotation matrices
            self.rot_mat_, self.views = rot_mat(self.phi0, self.phi1, self.phi_unit, 
                self.theta0, self.theta1, self.theta_unit, 
                sample_type=self.sample_type, rng=self.rng,
                nv=self.n_views)
            
            if self.n_views is None:
                self.n_views = len(self.views)
            print('# vps:', self.n_views)
            assert(self.n_views % self.v_batch == 0)

        # render 3d volume
        transmit = tf.exp(-tf.cumsum(d[:,::-1], axis=1)*self.transmit)
        d = tf.reduce_sum(d[:,::-1]*transmit, axis=1)
        d /= tf.reduce_max(d) # [0,1]

        # resize if needed 
        if abs(self.resize_scale - 1) > 1e-7:
            h = tf.to_int32(tf.multiply(float(self.resize_scale), tf.to_float(tf.shape(d)[1])))
            w = tf.to_int32(tf.multiply(float(self.resize_scale), tf.to_float(tf.shape(d)[2])))
            d = tf.image.resize_images(d, size=[h, w])

        # change the range of image to [0-255]
        self.d_img = tf.concat([d*255]*3, axis=-1) # [B,H,W,3]

        # plug-in to the pre-trained network
        imagenet_mean = 117.0
        d_preprocessed = self.d_img - imagenet_mean
        tf.import_graph_def(graph_def, {'input': d_preprocessed})
        self.layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        # print(self.layers)

    def _layer(self, layer):
        if 'input' in layer:
            return self.d_img

        if 'vgg' in self.model_path:
            return self.layers[layer]
        else:
            return self.graph.get_tensor_by_name("import/%s:0" % layer)

    def _gram_matrix(self, x):
        g_ = []
        for i in range(self.b_num):
            F = tf.reshape(x[i], (-1, x.shape[-1]))
            g = tf.matmul(tf.transpose(F), F)
            g_.append(g)
        return tf.stack(g_, axis=0)

    def _loss(self, params):
        self.content_loss = 0
        self.style_loss = 0
        self.total_loss = 0

        if self.w_content:
            feature = self._layer(self.content_layer) # assert only one layer
            if 'content_target' in params:
                self.content_feature = tf.placeholder(tf.float32)
                # self.content_loss -= tf.reduce_mean(feature*self.content_feature) # dot
                self.content_loss += tf.reduce_mean(tf.squared_difference(feature, 
                                               self.content_feature*self.w_content_amp))
            else:
                if self.content_channel:
                    self.content_loss -= tf.reduce_mean(feature[...,self.content_channel])
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,:self.content_channel]))
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,self.content_channel+1:]))
                else:
                    self.content_loss -= tf.reduce_mean(feature)

            self.total_loss += self.content_loss*self.w_content

        if self.w_style and 'style_target' in params:
            self.style_features = []
            self.style_denoms = []
            style_layers = self.style_layer.split(',')
            for style_layer in style_layers:
                feature = self._layer(style_layer)
                gram = self._gram_matrix(feature)
                f_shp = feature.shape
                style_feature = tf.placeholder(tf.float32, shape=f_shp)
                style_gram = self._gram_matrix(style_feature)

                style_denom = tf.placeholder(tf.float32, shape=1)
                self.style_loss += tf.reduce_sum(tf.squared_difference(gram, style_gram)) / style_denom
                self.style_features.append(style_feature)
                self.style_denoms.append(style_denom)

            self.total_loss += self.style_loss*self.w_style

    def _content_feature(self, content_target, content_shp):
        if abs(self.resize_scale - 1) > 1e-7:
            content_shp = [int(s*self.resize_scale) for s in content_shp]
        content_target_ = resize(content_target, content_shp)
        feature = self._layer(self.content_layer)
        feature_ = self.sess.run(feature, {self.d_img: [content_target_]*self.b_num})

        if self.top_k > 0:
            assert('softmax2_pre_activation' in self.content_layer)
            feature_k_ = self.sess.run(tf.nn.top_k(np.abs(feature_), k=self.top_k))
            for i in range(len(feature_)):
                exclude_idx = np.setdiff1d(np.arange(feature_.shape[1]), feature_k_.indices[i])
                feature_[i,exclude_idx] = 0
        
        return feature_

    def _style_feature(self, style_target, style_shp):
        style_mask = None
        if style_target.shape[-1] == 4:
            style_mask = style_target[...,-1] / 255
            style_target = style_target[...,:-1]

            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(style_target.astype(np.uint8))
            # plt.subplot(132)
            # plt.imshow(style_mask)
            # plt.subplot(133)
            # plt.imshow(np.stack([style_mask]*3, axis=-1)*(style_target/255))
            # plt.show()
        
        if abs(self.resize_scale - 1) > 1e-7:
            style_shp = [int(s*self.resize_scale) for s in style_shp]
        style_target_ = resize(style_target, style_shp)
        style_layers = self.style_layer.split(',')
        w_style_layers = self.w_style_layer.split(',')
        style_features = []
        style_denoms = []
        for style_layer, w_style_layer in zip(style_layers, w_style_layers):
            style_feature = self._layer(style_layer)
            style_feature_ = self.sess.run(style_feature, {self.d_img: [style_target_]*self.b_num})
            # style_gram = self._gram_matrix(style_feature)
            # style_gram_ = self.sess.run(style_gram, {self.d_img: [style_target_]})
            # plt.figure()
            # plt.imshow(style_gram_)
            # plt.show()

            f_shp = style_feature_.shape
            area = f_shp[1]*f_shp[2]
            nc = f_shp[3]
            denom = [4.0 * area**2 * nc**2 * 1e6 / float(w_style_layer)]
            if style_mask is not None:
                feature_mask = resize(style_mask, style_feature_.shape[1:-1])
                feature_mask = np.stack([feature_mask]*style_feature_.shape[-1], axis=-1)
                for i in range(self.b_num):
                    style_feature_[i] *= feature_mask
                    # plt.figure()
                    # # plt.subplot(121)
                    # plt.imshow(style_feature_[i,...,0])
                    # # plt.subplot(122)
                    # # plt.imshow(feature_mask)
                    # plt.show()
            style_features.append(style_feature_)
            style_denoms.append(denom)

        return style_features, style_denoms

    def _transport(self, g, v, a, b):
        if a < b:
            for i in range(a,b):
                g = self.sess.run(self.adv, {self.g: g, self.v: v[i,None]})
        elif a > b:
            for i in reversed(range(b,a)):
                g = self.sess.run(self.adv, {self.g: g, self.v: -v[i,None]})
        return g

    def run(self, params):
        # loss
        self._loss(params)

        # gradient
        g = tf.gradients(-self.total_loss, self.d_opt)[0]

        # laplacian gradient normalizer
        grad_norm = tffunc(np.float32)(partial(lap_normalize, 
            scale_n=self.lap_n, c=self.c, is_3d=True))

        d = params['d']
        if 'mask' in params:
            mask = params['mask']
            mask = np.stack([mask]*self.c, axis=-1)

        if 'v' in params:
            v = params['v']

        # settings for octave process
        oct_size = []
        hw = np.int32(d.shape)[1:]
        for _ in range(self.octave_n):
            oct_size.append(hw.copy())
            hw = np.int32(np.float32(hw)/self.octave_scale)
        print('input size for each octave', oct_size)

        d_shp = [self.num_frames] + [s for s in oct_size[-1]] + [self.c]
        d_opt_ = np.zeros(shape=d_shp, dtype=np.float32)

        # optimize
        loss_history = []
        d_opt_iter = []
        for octave in trange(self.octave_n):
            # octave process: scale-down for input
            if octave < self.octave_n-1:
                d_ = []
                for i in range(self.num_frames):
                    d_.append(resize(d[i], oct_size[-octave-1]))
                d_ = np.array(d_)

                if 'mask' in params:
                    mask_ = []
                    for i in range(self.num_frames):
                        m = resize(mask[i], oct_size[-octave-1])
                        mask_.append(m)

                if 'v' in params:
                    v_ = []
                    for i in range(self.num_frames-1):
                        v_.append(resize(v[i], oct_size[-octave-1]))
                    v_ = np.array(v_)
            else:
                d_ = d
                if 'mask' in params: mask_ = mask
                if 'v' in params: v_ = v

            if octave > 0:
                d_opt__ = []
                for i in range(self.num_frames):
                    d_opt__.append(resize(d_opt_[i], oct_size[-octave-1]))
                del d_opt_
                d_opt_ = np.array(d_opt__)
            
            feed = {}
            
            if 'content_target' in params:
                feed[self.content_feature] = self._content_feature(
                    params['content_target'], oct_size[-octave-1][1:])

            if 'style_target' in params:
                style_features, style_denoms = self._style_feature(
                    params['style_target'], oct_size[-octave-1][1:]
                )

                for i in range(len(self.style_features)):
                    feed[self.style_features[i]] = style_features[i]
                    feed[self.style_denoms[i]] = style_denoms[i]
            
            for step in trange(self.iter):
                g__ = []
                for t in trange(self.num_frames):
                    feed[self.d] = d_[t]
                    feed[self.d_opt] = d_opt_[t,None]

                    if self.rotate:
                        g_ = None
                        l_ = 0
                        for i in range(0, self.n_views, self.v_batch):
                            feed[self.rot_mat] = self.rot_mat_[i:i+self.v_batch]
                            g_vp, l_vp = self.sess.run([g, self.total_loss], feed)
                            if g_ is None:
                                g_ = g_vp
                            else:
                                g_ += g_vp
                            l_ += l_vp
                        l_ /= np.ceil(self.n_views/self.v_batch)

                        if not 'uniform' in self.sample_type:
                            self.rot_mat_, self.views = rot_mat(
                                self.phi0, self.phi1, self.phi_unit, 
                                self.theta0, self.theta1, self.theta_unit, 
                                sample_type=self.sample_type, rng=self.rng,
                                nv=self.n_views)
                    else:
                        g_, l_ = self.sess.run([g, self.total_loss], feed)
                        loss_history.append(l_)

                    g_ = denoise(g_, sigma=self.g_sigma)

                    if 'lr' in params:
                        lr = params['lr'][min(t, len(params['lr'])-1)]
                        g_[0] = grad_norm(g_[0]) * lr
                    else:
                        g_[0] = grad_norm(g_[0]) * self.lr
                    if 'mask' in params: g_[0] *= mask_[t]

                    g__.append(g_)

                if self.window_size > 1:
                    n = (self.window_size-1) // 2
                    for t in range(self.num_frames):
                        t0 = np.maximum(t - n, 0)
                        t1 = np.minimum(t + n, self.num_frames-1)
                        # print(t, t0, t1)
                        w = [1/(t1-t0+1)]*self.num_frames

                        g_ = g__[t].copy() * w[t]
                        for s in range(t0,t1+1):
                            if s == t: continue
                            g_ += self._transport(g__[s].copy(), v_, s, t) * w[s] # move s to t

                        d_opt_[t] += g_[0]
                        g__[t] = g_
                else:
                    for t in range(self.num_frames):
                        d_opt_[t] += g__[t][0]

                # to avoid resizing numerical error
                if 'mask' in params:
                    for t in range(self.num_frames):
                        d_opt_[t] *= np.ceil(mask_[t])

                if self.houdini and self.iter_seg > 0 and octave == self.octave_n-1:
                    if step % self.iter_seg == 0 and step > 0 and step < self.iter-1:
                        d_opt_iter.append(np.array(d_opt_, copy=True))
        
        # gather outputs
        result = {'l': loss_history}

        d_opt_iter = np.array(d_opt_iter)
        d_iter = []
        for i in range(d_opt_iter.shape[0]):
            d__ = []
            d_out_ = tf.identity(self.d_out)
            #feed_ = tf.identity(feed)
            for t in range(self.num_frames):
                feed[self.d_opt] = d_opt_iter[i, t, None]
                feed[self.d] = d[t]
                d__.append(self.sess.run(d_out_, feed)[0,...,0])
            d__ = np.array(d__)
            d_iter.append(d__)
        d_iter = np.array(d_iter)
        result['d_iter'] = d_iter
       
        d_ = []
        for t in range(self.num_frames):
            feed[self.d_opt] = d_opt_[t,None]
            feed[self.d] = d[t]
            d_.append(self.sess.run(self.d_out, feed)[0,...,0])
        d_ = np.array(d_)
        result['d'] = d_
        
        return result

##########
# Helper functions for Houdini Plugin
def help_send(data):
    if (is_py3()):
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.write(("|").encode())
    else:    
        sys.stdout.write(data)
        sys.stdout.write(("|").encode())

def help_receive():
    variable = ""
    next_byte = sys.stdin.read(1)
    while str(next_byte) != "|":
        variable = variable + next_byte
        next_byte = sys.stdin.read(1)
    return variable

def is_py3():
    return (sys.version_info > (3, 0))
##########


def stylize(args):
    # create a styler
    styler = Styler(vars(args))

    prepare_dirs_and_logger(args)

    # set file path format
    d_path_format = os.path.join(args.data_dir, 'd', '%03d.npz')
    v_path_format = os.path.join(args.data_dir, 'v', '%03d.npz')

    # directories for some visual results
    d_dir = os.path.join(args.log_dir, 'd') # original
    v_dir = os.path.join(args.log_dir, 'v') # velocity-mid slice
    r_dir = os.path.join(args.log_dir, 'r') # result
    for img_dir in [d_dir, v_dir, r_dir]:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    ########
    # Houdini plugin
    if args.houdini:
        control_data = bytes(1)
        control_stop = bytes(0)
        # First step (D and V Houdini): Load all data from stdin
        first_frame = True
        d_img_amount = []
        d_max = 0
        v_max = 0
        while True:
            data=sys.stdin.read(1)
            if (data == control_data and not is_py3()) or (bytes(int(data)) == control_data):
                info_dlen = help_receive()
                info_vlen = help_receive()
                info_frame = help_receive()
                if (is_py3()):
                    frame_d = pickle.loads(bytes(sys.stdin.read(int(info_dlen)), 'latin1'), encoding='latin1')
                else:
                    frame_d = pickle.loads(sys.stdin.read(int(info_dlen)))
                sys.stdin.read(1) # Consume
                if (is_py3()):
                    frame_v = pickle.loads(bytes(sys.stdin.read(int(info_vlen)), 'latin1'), encoding='latin1')
                else:
                    frame_v = pickle.loads(sys.stdin.read(int(info_vlen)))
                sys.stdin.read(1) # Consume
                if frame_d.max() > d_max: d_max = frame_d.max()
                if np.max(np.abs(frame_v)) > v_max: v_max = frame_v.max()

                transmit = np.exp(-np.cumsum(frame_d[::-1], axis=0)*args.transmit)
                d_img = np.sum(frame_d[::-1]*transmit, axis=0)
                d_img /= d_img.max()
                # Optional Step: Save original slices
                d_xy, d_xz, d_yz = np.array(frame_d, copy=True), np.array(frame_d, copy=True), np.array(frame_d, copy=True)
                mid_xy, mid_xz, mid_yz = int(d_xy.shape[0]/2), int(d_xz.shape[1]/2), int(d_yz.shape[2]/2)
                d_xy, d_xz, d_yz = d_xy[mid_xy, ...], d_xz[:, mid_xz, ...], d_yz[..., mid_yz]
                if d_xy.max() > 0: d_xy /= d_xy.max()
                if d_xz.max() > 0: d_xz /= d_xz.max()
                if d_yz.max() > 0: d_yz /= d_yz.max()
                i_xy, i_xz, i_yz = d_xy*255, d_xz*255, d_yz*255
                i_xy, i_xz, i_yz = Image.fromarray(i_xy.astype(np.uint8)), Image.fromarray(i_xz.astype(np.uint8)), Image.fromarray(i_yz.astype(np.uint8))
                i_xy_path = os.path.join(d_dir, 'd_xy' + ('%03d.png' % int(info_frame)))
                i_xz_path = os.path.join(d_dir, 'd_xz' + ('%03d.png' % int(info_frame)))
                i_yz_path = os.path.join(d_dir, 'd_yz' + ('%03d.png' % int(info_frame)))
                i_xy.save(i_xy_path)
                i_xz.save(i_xz_path)
                i_yz.save(i_yz_path)

                v_xy, v_xz, v_yz = np.array(frame_v, copy=True), np.array(frame_v, copy=True), np.array(frame_v, copy=True)
                mid_xy, mid_xz, mid_yz = int(v_xy.shape[0]/2), int(v_xz.shape[1]/2), int(v_yz.shape[2]/2)
                v_xy = (v_xy[mid_xy,...,0]**2 + v_xy[mid_xy,...,1]**2 + v_xy[mid_xy,...,2]**2)**0.5
                v_xz = (v_xz[:, mid_xz, ..., 0]**2 + v_xz[:, mid_xz, ..., 1]**2 + v_xz[:, mid_xz, ..., 2]**2)**0.5
                v_yz = (v_yz[..., mid_yz, 0]**2 + v_yz[..., mid_yz, 1]**2  + v_yz[..., mid_yz, 2]**2)**0.5
                if v_xy.max() > 0: v_xy /= (v_xy.max() + 1e-7)
                if v_xz.max() > 0: v_xz /= (v_xz.max() + 1e-7)
                if v_yz.max() > 0: v_yz /= (v_yz.max() + 1e-7)
                i_xy, i_xz, i_yz = np.uint8(plt.cm.viridis(v_xy)*255), np.uint8(plt.cm.viridis(v_xz)*255), np.uint8(plt.cm.viridis(v_yz)*255)
                i_xy, i_xz, i_yz = Image.fromarray(i_xy), Image.fromarray(i_xz), Image.fromarray(i_yz)
                i_xy_path = os.path.join(v_dir, 'v_xy' + ('%03d.png' % int(info_frame)))
                i_xz_path = os.path.join(v_dir, 'v_xz' + ('%03d.png' % int(info_frame)))
                i_yz_path = os.path.join(v_dir, 'v_yz' + ('%03d.png' % int(info_frame)))
                i_xy.save(i_xy_path)
                i_xz.save(i_xz_path)
                i_yz.save(i_yz_path)

                t_xy, t_xz, t_yz = np.array(transmit, copy=True), np.array(transmit, copy=True), np.array(transmit, copy=True)
                mid_xy, mid_xz, mid_yz = int(t_xy.shape[0]/2), int(t_xz.shape[1]/2), int(t_yz.shape[2]/2)
                t_xy, t_xz, t_yz = t_xy[mid_xy, ...], t_xz[:, mid_xz, ...], t_yz[..., mid_yz]
                if t_xy.max() > 0: t_xy /= t_xy.max()
                if t_xz.max() > 0: t_xz /= t_xz.max()
                if t_yz.max() > 0: t_yz /= t_yz.max()
                t_xy, t_xz, t_yz = t_xy*255, t_xz*255, t_yz*255
                t_xy, t_xz, t_yz = Image.fromarray(t_xy.astype(np.uint8)), Image.fromarray(t_xz.astype(np.uint8)), Image.fromarray(t_yz.astype(np.uint8))
                t_xy_path = os.path.join(d_dir, 'txy' + ('%03d.png' % int(info_frame)))
                t_xz_path = os.path.join(d_dir, 'txz' + ('%03d.png' % int(info_frame)))
                t_yz_path = os.path.join(d_dir, 'tyz' + ('%03d.png' % int(info_frame)))
                t_xy.save(t_xy_path)
                t_xz.save(t_xz_path)
                t_yz.save(t_yz_path)

                d_img_amount.append(np.sum(d_img))
                if (args.single_frame):
                    assert(first_frame)
                    d = np.array([frame_d])
                    v = np.array([frame_v])
                    first_frame = False
                else: 
                    if first_frame:
                        first_frame = False
                        d_dim = frame_d.shape
                        v_dim = frame_v.shape
                        d = np.empty([args.num_frames, d_dim[0], d_dim[1], d_dim[2]])
                        v = np.empty([args.num_frames, v_dim[0], v_dim[1], v_dim[2], v_dim[3]])
                    d[int(info_frame) - args.target_frame, ...] = frame_d   
                    v[int(info_frame) - args.target_frame, ...] = frame_v
                print("Added frame: " + info_frame)
            elif (data == control_stop and not is_py3()) or (bytes(int(data)) == control_stop):
                # Second step: Normalize the densities and velocities
                d /= d_max
                v /= v_max
                print("Begin Stylization")
                break
            else:
                print("ERROR: Invalid control byte sent")
                print(data)
                sys.exit(0)

    else:
        d_path = d_path_format % args.target_frame

        print('load density fields')
        d = []
        d_img_amount = []
        for i in trange(args.num_frames):
            d_path = d_path_format % (args.target_frame+i)
            with np.load(d_path) as data:
                d_ = data['x'][:,::-1]

            if abs(args.scale - 1) > 1e-7:
                hw = [int(s*args.scale) for s in hw]
                d_ = resize(d_, hw)

            # save original density image
            transmit = np.exp(-np.cumsum(d_[::-1], axis=0)*args.transmit)
            d_img = np.sum(d_[::-1]*transmit, axis=0)
            d_img /= d_img.max()
            im = d_img*255
            im = Image.fromarray(im.astype(np.uint8))
            im_path = os.path.join(d_dir, '%03d.png' % (args.target_frame+i))
            im.save(im_path)

            d_img_amount.append(np.sum(d_img))
            d.append(d_)
        
        d = np.array(d)
        d_shp = d.shape[1:] # zyx -> dhw

        print('load velocity fields')
        v_ = []
        for i in trange(args.num_frames-1):
            v_path = v_path_format % (args.target_frame+i)

            with np.load(v_path) as data:
                v = data['x']
                vx = np.dstack((v,np.zeros((v.shape[0],v.shape[1],1,v.shape[3]))))
                vx = (vx[:,:,1:,0] + vx[:,:,:-1,0]) * 0.5
                vy = np.hstack((v,np.zeros((v.shape[0],1,v.shape[2],v.shape[3]))))
                vy = (vy[:,1:,:,1] + vy[:,:-1,:,1]) * 0.5
                vz = np.vstack((v,np.zeros((1,v.shape[1],v.shape[2],v.shape[3]))))
                vz = (vz[1:,:,:,2] + vz[:-1,:,:,2]) * 0.5
                v = np.stack([vx,vy,vz], axis=-1)
                v = v[:,::-1]
            if v.shape[:-1] != d_shp: v = resize(v, d_shp)

            # save the middle slice of velocity field
            z_mid = int(v.shape[0]/2)
            m = (v[z_mid,...,0]**2 + v[z_mid,...,1]**2 + v[z_mid,...,2]**2)**0.5
            m /= (m.max() + 1e-7)
            # v_max = np.abs(v).max()
            # v_img = ((v / v_max + 1) / 2)[z_mid]
            im = np.uint8(plt.cm.viridis(m)*255)
            im = Image.fromarray(im)
            im_path = os.path.join(v_dir, '%03d.png' % (args.target_frame+i))
            im.save(im_path)

            time_step = 0.5
            vx = v[...,0] / time_step / v.shape[2] * (args.scale*0.5)
            vy = -v[...,1] / time_step / v.shape[1] * (args.scale*0.5)
            vz = v[...,2] / time_step / v.shape[0] * (args.scale*0.5)
            v = np.stack([vz,vy,vx], axis=-1)
            v_.append(v)
        
        v = np.array(v_)

    params = {'d': d}
    params['v'] = v
    d_shp = d.shape[1:] # zyx -> dhw
    
    # set learning rate depending on the amount of density
    d_amount = np.sum(d, axis=(1,2,3))
    d_img_amount = np.array(d_img_amount)
    d_img_amount /= d_img_amount.max()
    params['lr'] = d_img_amount*args.lr
    

    # mask
    if args.mask:
        params['mask'] = denoise(d, args.g_sigma)

    # load a content target image
    if args.content_target:
        content_target = np.float32(Image.open(args.content_target))
        # remove alpha channel
        if content_target.shape[-1] == 4:
            content_target = content_target[...,:-1]
        
        # crop
        ratio = d_shp[2] / d_shp[1] # x/y
        content_target = crop_ratio(content_target, ratio)

        # range is still [0-255]
        params['content_target'] = content_target
        # plt.figure()
        # plt.imshow(content_target/255)
        # plt.show()

    if args.style_target:
        style_target = np.float32(Image.open(args.style_target))
        
        # crop
        ratio = d_shp[2] / d_shp[1] # x/y
        style_target = crop_ratio(style_target, ratio)
            
        # range is still [0-255]
        params['style_target'] = style_target
        # plt.figure()
        # plt.imshow(style_target/255)
        # plt.show()

    #########
    # stylize
    result = styler.run(params)
    d_sty, loss, d_iter = result['d'], result['l'], result['d_iter']

    if args.houdini:
        print("complete") # !!!! Houdini will consume this message !!!!
        # Eigth Step (Houdini): Denormalize results
        if d_iter.size == 0:
            d_iter = np.array([d_sty])
        else:
            d_iter = np.vstack([d_iter, [d_sty]])
        d_iter *= d_max
        
        inters = []
        if args.iter_seg > 0:
            for i in range(args.iter):
                if i % args.iter_seg == 0 and i != 0:
                    inters.append(i)
        inters.append(args.iter)
        assert(len(inters) == d_iter.shape[0]), ("Prog ERROR: Incorrect definition of iterables")
        
        # d_iter = (iter, frame, D, H, W)
        for it, d_iter_ in enumerate(d_iter):
            for i, d_sty_ in enumerate(d_iter_):
                # Optional Step: Save returned results
                transmit = np.exp(-np.cumsum(d_sty_[::-1], axis=0)*args.transmit)
                d_sty_img = np.sum(d_sty_[::-1]*transmit, axis=0)
                d_sty_img /= d_sty_img.max()
                im = d_sty_img*255
                im = Image.fromarray(im.astype(np.uint8))
                im_path = os.path.join(r_dir, ('i%03d_' % (inters[it])) + ('%03d.png' % (args.target_frame+i)) )
                im.save(im_path)

                # Ninth Step (Houdini): Return results
                frame_r = np.array(d_sty_, copy=True, dtype = np.float32)
                frame_r = frame_r[:,::-1]
                if (is_py3()):
                    data_r = pickle.dumps(frame_r, 2)
                    info_r = str(len(data_r)).encode()
                    sys.stdout.write(str(len(control_data)))
                    help_send(str(inters[it]).encode())
                    help_send(str(int(args.target_frame+i)).encode())
                    help_send(info_r)
                    help_send(data_r)
                else:
                # NOTE: Python 2.7 modified to use tensorflow cannot use pickle dumps
                # We will have to send data back via a string
                    frame_r = np.around(frame_r, decimals = 2)
                    util_dim = frame_r.shape
                    frame_r = frame_r.flatten()
                    r_string = " ".join(str(elem) for elem in frame_r)

                    sys.stdout.write(control_data)
                    help_send(str(inters[it]).encode())
                    help_send(str(int(args.target_frame+i)).encode())
                    help_send(str(int(util_dim[2])).encode())
                    help_send(str(int(util_dim[1])).encode())
                    help_send(str(int(util_dim[0])).encode())
                    print(r_string)
        if (is_py3()):
            sys.stdout.write(str(len(control_stop)))
        else:    
            sys.stdout.write(control_stop)
    else:
        for i, d_sty_ in enumerate(d_sty):
            # save stylized density
            d_path = os.path.join(args.log_dir, '%03d.npz' % (args.target_frame+i))
            np.savez_compressed(d_path, x=d_sty_[:,::-1])

            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(d_sty_[100-20]) # back
            # plt.subplot(122)
            # plt.imshow(d_sty_[100+20]) # front
            # plt.show()
            
            # save image
            transmit = np.exp(-np.cumsum(d_sty_[::-1], axis=0)*args.transmit)
            d_sty_img = np.sum(d_sty_[::-1]*transmit, axis=0)
            d_sty_img /= d_sty_img.max()
                
            im = d_sty_img*255
            im = Image.fromarray(im.astype(np.uint8))
            im_path = os.path.join(r_dir, '%03d.png' % (args.target_frame+i))
            im.save(im_path)

if __name__ == '__main__':
    args = parser.parse_args()
    stylize(args)