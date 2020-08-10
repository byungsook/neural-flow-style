#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange
from util import *
from transform import p2g
import vgg
from styler_base import StylerBase

class Styler(StylerBase):
    def __init__(self, self_dict):
        StylerBase.__init__(self, self_dict)

        # particle position        
        # shape: [N,2], scale: [0,1]
        p = []
        p_shp = [None,2]
        self.p = []
        
        # particle density shape
        r_shp = [None,1]
        self.r = []

        # particle color shape
        c_shp = [None,3]
        self.c = []

        # output and density field
        d = []
        d_gray = []

        self.opt_init = []
        self.opt_ph = []
        self.opt = []

        self.res = tf.compat.v1.placeholder(tf.int32, [2], name='resolution')

        for i in range(self.batch_size):
            # particle position, [N,2]
            p_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=p_shp, name='p%d' % i)
            self.p.append(p_)
            p_ = tf.expand_dims(p_, axis=0) # [1,N,2]
            p.append(p_[0])

            # particle density, [N,1]
            r_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=r_shp, name='r%d' % i)
            self.r.append(r_)
            r_ = tf.expand_dims(r_, axis=0) # [1,N,1]
            
            # position-based (SPH) density field estimation
            d_gray_ = p2g(p_, self.domain, self.res, self.radius, self.rest_density, self.nsize, support=self.support, clip=self.clip) # [B,N,2] -> [B,H,W,1]
            d_gray_ /= self.rest_density # normalize density
            d_gray.append(d_gray_)

            # particle color, [N,3]
            opt_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=c_shp, name='c_opt_ph%d' % i)
            self.opt_ph.append(opt_ph)
            opt_var = tf.Variable(opt_ph, validate_shape=False, name='c_opt%d' % i)
            self.opt.append(opt_var)
            opt_var_ = tf.reshape(opt_var, tf.shape(opt_ph))
            opt_var_ = tf.expand_dims(opt_var_, axis=0)
            
            # clip particle color
            c_ = tf.clip_by_value(opt_var_, 0, 1)

            # mask color
            self.c.append(c_[0]*tf.clip_by_value(r_[0]/self.rest_density, 0, 1))

            # position-based (SPH) color field estimation
            d_ = p2g(p_, self.domain, self.res, self.radius, self.rest_density, self.nsize, support=self.support, clip=self.clip,
                        pc=c_, pd=r_) # [B,N,2] -> [B,H,W,3]
            
            d.append(d_)

        self.opt_init = tf.compat.v1.initializers.variables(self.opt)

        # particle position
        self.p_out = p # [N,2]*B
        
        # estimated color fields
        d = tf.concat(d, axis=0) # [B,H,W,3]

        # value clipping for rendering
        d = tf.clip_by_value(d, 0, 1)

        # estimated density fields for masking
        d_gray = tf.concat(d_gray, axis=0) # [B,H,W,1]

        # clamp density field [0,1]
        d_gray = tf.clip_by_value(d_gray, 0, 1)

        # mask for style features
        self.d_gray = d_gray

        # stylized result
        self.d_out = d*d_gray # [B,H,W,3]

        self._plugin_to_loss_net(d)

    def render_test(self, params):
        feed = {}
        feed[self.res] = self.resolution
        
        for i in range(self.batch_size):
            feed[self.p[i]] = params['p'][i]
            feed[self.r[i]] = params['r'][i]
            n = params['p'][i].shape[0]

            # feed[self.opt_ph[i]] = np.ones([n,3])
            c_init_shp = [n,3]
            c_init = self.rng.uniform(-5,5, c_init_shp).astype(np.float32)
            c_init += np.array([vgg._R_MEAN, vgg._G_MEAN, vgg._B_MEAN])
            feed[self.opt_ph[i]] = c_init/255

        self.sess.run(self.opt_init, feed)
        p_out, d_out, d_gray = self.sess.run([self.p_out, self.d_out, self.d_gray], feed)
        plt.subplot(121)
        plt.imshow(d_out[0])
        plt.subplot(122)
        plt.imshow(d_gray[0,...,0])
        plt.show()

        for i, p in enumerate(p_out):
            p[:,0] = p[:,0]*self.domain[0]
            p[:,1] = p[:,1]*self.domain[1]
            p_out[i] = np.stack([p[:,1],p[:,0]], axis=-1)
        v_ = None
        bbox = [
            [0,0,-1],
            [self.domain[1],self.domain[0],1],
            ]
        draw_pt(p_out, v_, bbox=bbox)
        return

        # save to image
        for t in trange(0,self.num_frames,self.batch_size):
            if t == 0:
                n = params['p'][0].shape[0]
                from matplotlib import cm
                c = cm.plasma(np.linspace(0,1,n))[...,:-1]
                
            for i in range(self.batch_size):
                feed[self.p[i]] = params['p'][t+i]
                feed[self.r[i]] = params['r'][t+i]
                if 'p' in self.target_field:
                    feed[self.opt_ph[i]] = np.zeros([n,2])
                if 'c' in self.target_field:
                    feed[self.opt_ph[i]] = c
                
            self.sess.run(self.opt_init, feed)    
            d_out = self.sess.run(self.d_out, feed)
            if d_out.shape[-1] == 1:
                d_out = d_out[...,0] # [B,H,W]
            # plt.imshow(d_out[0])
            # plt.show()
            for i in range(self.batch_size):                
                im = Image.fromarray((d_out[i]*255).astype(np.uint8))
                d_path = os.path.join(self.log_dir, '%03d.png' % (t+i))
                im.save(d_path)

    def run(self, params):
        # loss
        self._loss(params)

        # optimizer
        self.opt_lr = tf.compat.v1.placeholder(tf.float32)

        # settings for octave process
        oct_size = []
        hw = np.array(self.resolution)
        for _ in range(self.octave_n):
            oct_size.append(hw)
            hw = (hw//self.octave_scale).astype(np.int)
        oct_size.reverse()
        print('input size for each octave', oct_size)

        p = params['p']
        r = params['r']

        g_opt = []
        n = p[0].shape[0] # n is fixed
        # # same noise
        # c_opt_shp = [n, 3]
        # different noise
        c_opt_shp = [self.num_frames, n, 3]
        c_opt = self.rng.uniform(-5,5, c_opt_shp).astype(np.float32)
        c_opt += np.array([vgg._R_MEAN, vgg._G_MEAN, vgg._B_MEAN])
        c_opt /= 255 # [0,1]
        for i in range(self.num_frames):
            # # same noise
            # c_opt.append(c_opt)
            # different noise
            g_opt.append(c_opt[i])

        # optimize
        loss_history = []
        d_intm = []
        opt_ = {}
        for octave in trange(self.octave_n, desc='octave'):
            loss_history_o = []
            d_intm_o = []

            feed = {}
            feed[self.res] = oct_size[octave]
            if self.content_img is not None:
                feed[self.content_feature] = self._content_feature(
                    self.content_img, oct_size[octave])

            if self.style_img is not None:
                style_features = self._style_feature(
                    self.style_img, oct_size[octave])
                
                for i in range(len(self.style_features)):
                    feed[self.style_features[i]] = style_features[i]

                if self.w_hist > 0:
                    hist_features = self._hist_feature(
                        self.style_img, oct_size[octave])
                    
                    for i in range(len(self.hist_features)):
                        feed[self.hist_features[i]] = hist_features[i]

            if type(self.lr) == list:
                lr = self.lr[octave]
            else:
                lr = self.lr

            # optimizer list for each batch
            for step in trange(self.iter,desc='iter'):
                g_tmp = [None]*self.num_frames

                for t in range(0,self.num_frames,self.batch_size):
                    for i in range(self.batch_size):
                        feed[self.p[i]] = p[t+i]
                        feed[self.r[i]] = r[t+i]
                        feed[self.opt_ph[i]] = g_opt[t+i]
                    
                    # assign g_opt to self.opt through self.opt_ph
                    self.sess.run(self.opt_init, feed)

                    feed[self.opt_lr] = lr
                    opt_id = t//self.frames_per_opt
                    # opt_id = self.rng.randint(num_opt)
                    if opt_id in opt_:
                        train_op = opt_[opt_id]
                    else:
                        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.opt_lr)
                        train_op = opt.minimize(self.total_loss, var_list=self.opt)
                        self.sess.run(tf.compat.v1.variables_initializer(opt.variables()), feed)
                        opt_[opt_id] = train_op

                    # optimize
                    _, l_ = self.sess.run([train_op, self.total_loss], feed)
                    loss_history_o.append(l_)

                    g_opt_ = self.sess.run(self.opt, feed)
                    for i in range(self.batch_size):
                        g_tmp[t+i] = np.nan_to_num(g_opt_[i]) - g_opt[t+i]

                    if step == self.iter-1 and octave < self.octave_n-1: # True or 
                        d_intm_ = self.sess.run(self.d_out, feed)
                        d_intm_o.append((d_intm_*255).astype(np.uint8))

                        # ## debug
                        # d_gray = self.sess.run(self.d_gray, feed)
                        # plt.subplot(121)
                        # plt.imshow(d_intm_[0,...])
                        # plt.subplot(122)
                        # plt.imshow(d_gray[0,...,0])
                        # plt.show()

                #########
                # gradient alignment
                if self.window_sigma > 0 and self.num_frames > 1:
                    g_tmp = denoise(g_tmp, sigma=(self.window_sigma,0,0))

                for t in range(self.num_frames):
                    g_opt[t] += g_tmp[t]

            loss_history.append(loss_history_o)
            if octave < self.octave_n-1:
                d_intm.append(np.concatenate(d_intm_o, axis=0))

        # gather outputs
        result = {
            'l': loss_history, 'd_intm': d_intm,
            }

        # final inference
        c_sty = [None]*self.num_frames
        d_sty = [None]*self.num_frames
        for t in range(0,self.num_frames,self.batch_size):
            for i in range(self.batch_size):
                feed[self.p[i]] = p[t+i]
                feed[self.r[i]] = r[t+i]
                feed[self.opt_ph[i]] = g_opt[t+i]

            self.sess.run(self.opt_init, feed)            
            p_, d_ = self.sess.run([self.p_out, self.d_out], feed)
            c_ = self.sess.run(self.c, feed)
            
            for i in range(self.batch_size):
                c_sty[t+i] = c_[i]

            d_ = (d_*255).astype(np.uint8)
            d_sty[t:t+self.batch_size] = d_

        result['c'] = c_sty
        result['d'] = np.array(d_sty)

        return result