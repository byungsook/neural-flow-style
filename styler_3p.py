#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange
from util import *
from transform import p2g, p2g_wavg, rotate, rot_mat
import vgg
from styler_base import StylerBase

class Styler(StylerBase):
    def __init__(self, self_dict):
        StylerBase.__init__(self, self_dict)

        # particle position
        # shape: [N,3], scale: [0,1]
        p = []
        p_shp = [None,3]
        self.p = [] # input
        self.v = [] # style

        # particle density, [N,nk]
        r_shp = [None,self.num_kernels]
        self.r = [] # input
        self.d = [] # style

        # output
        d = []
        d_gray = []

        pressure = []

        self.opt_init = []
        self.opt_ph = []
        self.opt = []

        self.res = tf.compat.v1.placeholder(tf.int32, [3], name='resolution')

        for i in range(self.batch_size):
            # particle position, [N,3]
            p_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=p_shp, name='p%d' % i)
            self.p.append(p_)
            p_ = tf.expand_dims(p_, axis=0) # [1,N,3]

            # particle velocity, [N,3]
            if 'p' in self.target_field:
                p_opt_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=p_shp, name='p_opt_ph%d' % i)
                self.opt_ph.append(p_opt_ph)
                p_opt = tf.Variable(p_opt_ph, validate_shape=False, name='p_opt%d' % i)
                self.opt.append(p_opt)
                p_opt_ = tf.reshape(p_opt, tf.shape(p_opt_ph))
                p_opt_ = tf.expand_dims(p_opt_, axis=0)
                v_ = p_opt_
                self.v.append(v_[0])
                p_ += v_

            p.append(p_[0])

            # particle density, [N,nk]
            if 'd' in self.target_field:
                r_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=r_shp, name='r%d' % i)
                self.r.append(r_)
                r_ = tf.expand_dims(r_, axis=0) # [1,N,nk]

                r_opt_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=r_shp, name='r_opt_ph')
                self.opt_ph.append(r_opt_ph)
                r_opt = tf.Variable(r_opt_ph, validate_shape=False, name='r_opt')
                self.opt.append(r_opt)
                r_opt_ = tf.reshape(r_opt, tf.shape(r_opt_ph))
                r_opt_ = tf.expand_dims(r_opt_, axis=0) # [1,N,nk]
                r_opt_ = tf.clip_by_value(r_opt_, -1, 1) #### necessary!
                self.d.append(r_opt_[0])
                r_ += r_opt_

                # weighted avg. density estimation
                for k in range(self.num_kernels):
                    factor = self.kernel_scale**k
                    support = self.support/factor
                    r_k = tf.expand_dims(r_[...,k], axis=-1)
                    d_hat = p2g_wavg(p_, r_k, self.domain, self.res, self.radius, self.nsize, kernel='cubic', support=support, clip=self.clip, is_2d=False)
                    if k == 0:
                        d_ = d_hat
                    else:
                        d_ += d_hat
            else:            
                # position-based (SPH) density field estimation
                d_ = p2g(p_, self.domain, self.res, self.radius, self.rest_density, self.nsize, support=self.support, clip=self.clip, is_2d=False) # [B,N,3] -> [B,D,H,W,1]
                d_ /= self.rest_density # normalize density
            
            d.append(d_)

            # pressure estimation
            if self.w_pressure > 0 and 'p' in self.target_field:
                pressure_ = tf.where(d_>0, d_-1, tf.zeros_like(d_))
                pressure.append(pressure_)

        self.opt_init = tf.compat.v1.initializers.variables(self.opt)

        # stylized (advected) particles
        self.p_out = p # [N,3]*B
        
        # estimated density fields
        d = tf.concat(d, axis=0) # [B,D,H,W,1]

        if self.w_pressure > 0 and 'p' in self.target_field:
            pressure = tf.concat(pressure, axis=0) # [B,D,H,W,1]
            self.pressure = pressure
        
        if self.k > 0:
            # smoothing density for density optimization
            k = []
            k1 = np.float32([1,self.k,1])
            k2 = np.outer(k1, k1)
            for i in k1:
                k.append(k2*i)
            k = np.array(k)
            k = k[:,:,:,None,None]/k.sum()
            d = tf.nn.conv3d(d, k, [1,1,1,1,1], 'SAME')

        # value clipping for rendering
        # d = tf.clip_by_value(d, 0, 1)
        d = tf.maximum(d, 0)

        # stylized result
        self.d_out = d # [B,D,H,W,1]
        
        ####
        # rotate 3d smoke for rendering
        if self.rotate:
            d, self.rot_mat = rotate(d) # [B,D,H,W,1] or [B,D,H,W,4]
            self.d_out_rot = d

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
        if self.render_liquid:
            # d = tf.reduce_max(d, axis=1) # [B,H,W,1]
            transmit = tf.exp(-tf.cumsum(d[:,::-1], axis=1)*self.transmit)
            self.d_trans = transmit
            d = 1 - transmit[:,-1] # [B,H,W,1], [0,1]
            # d = (1 - transmit[:,-1])*np.array([0.26, 0.5, 0.75]) + transmit[:,-1]*np.array([1, 1, 1]) # [B,H,W,1], [0,1]
        else:
            transmit = tf.exp(-tf.cumsum(d[:,::-1], axis=1)*self.transmit)[:,::-1]
            d *= transmit
            d = tf.reduce_sum(d, axis=1) # [B,H,W,1] or [B,H,W,3]
            d /= tf.reduce_max(d) # [B,H,W,1], [0,1]

        # mask for style features
        self.d_gray = d # [B,H,W,1]
        ####

        self._plugin_to_loss_net(d)

    def render_test(self, params):
        feed = {}
        feed[self.res] = self.resolution

        if self.rotate:
            feed[self.rot_mat] = self.rot_mat_[:self.v_batch]
        
        for i in range(self.batch_size):
            feed[self.p[i]] = params['p'][i]
            n = params['p'][i].shape[0]
            if 'p' in self.target_field:
                feed[self.opt_ph[i]] = np.zeros([n,3])
            if 'd' in self.target_field:
                feed[self.r[i]] = params['r'][i]
                feed[self.opt_ph[i]] = np.zeros([n,self.num_kernels])

        self.sess.run(self.opt_init, feed)
        p_out, d_img, d_gray = self.sess.run([self.p_out, self.d_img, self.d_gray], feed)
        plt.subplot(121)
        plt.imshow(d_img[0].astype(np.uint8))
        plt.subplot(122)
        plt.imshow(d_gray[0,...,0])
        plt.show()

        for i, p in enumerate(p_out):
            p[:,0] = p[:,0]*self.domain[0]
            p[:,1] = p[:,1]*self.domain[1]
            p[:,2] = p[:,2]*self.domain[2]
            p_out[i] = np.stack([p[:,2],p[:,1],p[:,0]], axis=-1)
        v_ = None
        bbox = [
            [0,0,0],
            [self.domain[2],self.domain[1],self.domain[0]],
            ]
        draw_pt(p_out, pv=v_, bbox=bbox, is_2d=False)

        feed = {}
        feed[self.res] = self.resolution
        if self.rotate:
            feed[self.rot_mat] = [np.identity(3)]*self.batch_size
        
        # save to image
        for t in trange(0,self.num_frames,self.batch_size):
            if t == 0:
                n = params['p'][0].shape[0]
                
            for i in range(self.batch_size):
                feed[self.p[i]] = params['p'][t+i]
                if 'p' in self.target_field:
                    feed[self.opt_ph[i]] = np.zeros([n,3])
                if 'd' in self.target_field:
                    feed[self.r[i]] = params['r'][t+i]
                    feed[self.opt_ph[i]] = np.zeros([n,self.num_kernels])
                
            self.sess.run(self.opt_init, feed)
            d_out = self.sess.run(self.d_img, feed)
            # plt.imshow(d_out[0])
            # plt.show()
            for i in range(self.batch_size):
                im = Image.fromarray(d_out[i].astype(np.uint8))
                d_path = os.path.join(self.log_dir, '%03d.png' % (t+i))
                im.save(d_path)

    def run(self, params):
        # loss
        self._loss(params)

        # optimizer
        self.opt_lr = tf.compat.v1.placeholder(tf.float32)

        # adaptive learning rate per octave
        if abs(self.lr_scale - 1) > 1e-7:
            self.lr = [self.lr/self.lr_scale**i for i in range(self.octave_n)]

        # settings for octave process
        oct_size = []
        dhw = np.array(self.resolution)
        for _ in range(self.octave_n):
            oct_size.append(dhw)
            dhw = (dhw//self.octave_scale).astype(np.int)
        oct_size.reverse()
        print('input size for each octave', oct_size)

        p = params['p']
        
        g_opt = []
        if 'p' in self.target_field:
            for i in range(self.num_frames):
                n = p[i].shape[0]
                p_opt_shp = [n, 3]
                p_opt = np.zeros(shape=p_opt_shp, dtype=np.float32)
                g_opt.append(p_opt)

        if 'd' in self.target_field:
            r = params['r']
            for i in range(self.num_frames):
                n = p[i].shape[0]
                r_opt_shp = [n, self.num_kernels]
                r_opt_ = np.zeros(shape=r_opt_shp, dtype=np.float32)
                g_opt.append(r_opt_)

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
                    self.content_img, oct_size[octave][1:])

            if self.style_img is not None:
                style_features = self._style_feature(
                    self.style_img, oct_size[octave][1:])
                
                for i in range(len(self.style_features)):
                    feed[self.style_features[i]] = style_features[i]

                if self.w_hist > 0:
                    hist_features = self._hist_feature(
                        self.style_img, oct_size[octave][1:])
                    
                    for i in range(len(self.hist_features)):
                        feed[self.hist_features[i]] = hist_features[i]

            if type(self.lr) == list:
                lr = self.lr[octave]
            else:
                lr = self.lr

            # optimizer list for each batch
            for step in trange(self.iter,desc='iter'):
                g_tmp = [None]*self.num_frames

                for t in range(0,self.num_frames,self.batch_size*self.interp):
                    for i in range(self.batch_size):
                        feed[self.p[i]] = p[t+i*self.interp]
                        feed[self.opt_ph[i]] = g_opt[t+i*self.interp]
                        if 'd' in self.target_field:
                            feed[self.r[i]] = r[t+i*self.interp]
                    
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
                    if self.rotate:
                        g_opt_ = None
                        l_ = []
                        for i in range(0, self.n_views, self.v_batch):
                            feed[self.rot_mat] = self.rot_mat_[i:i+self.v_batch]
                            _, l_vp = self.sess.run([train_op, self.total_loss], feed)
                            l_.append(l_vp)

                            g_opt_i = self.sess.run(self.opt, feed)
                            
                            if i == 0:
                                g_opt_ = np.nan_to_num(g_opt_i)
                            else:
                                for j in range(self.batch_size):
                                    g_opt_[j] += np.nan_to_num(g_opt_i[j])
                        
                        loss_history_o.append(np.mean(l_))

                        if not 'uniform' in self.sample_type:
                            self.rot_mat_, self.views = rot_mat(
                                self.phi0, self.phi1, self.phi_unit, 
                                self.theta0, self.theta1, self.theta_unit, 
                                sample_type=self.sample_type, rng=self.rng,
                                nv=self.n_views)

                        for i in range(self.batch_size):
                            g_opt_[i] /= (self.n_views/self.v_batch)
                    else:
                        _, l_ = self.sess.run([train_op, self.total_loss], feed)
                        loss_history_o.append(l_)

                        g_opt_ = self.sess.run(self.opt, feed)
                        
                    for i in range(self.batch_size):
                        g_tmp[t+i*self.interp] = np.nan_to_num(g_opt_[i]) - g_opt[t+i*self.interp]
                        if 'd' in self.target_field:
                            # masking by original density
                            g_tmp[t+i*self.interp] *= r[t+i*self.interp][...,0,None]

                    if step == self.iter-1 and octave < self.octave_n-1: # True or 
                        if self.rotate:
                            feed[self.rot_mat] = [np.identity(3)]*self.batch_size

                        d_intm_ = self.sess.run(self.d_img, feed)
                        d_intm_o.append(d_intm_.astype(np.uint8))

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
                    g_tmp[:self.num_frames:self.interp] = denoise(g_tmp[:self.num_frames:self.interp], sigma=(self.window_sigma,0,0))

                for t in range(0,self.num_frames,self.interp):
                    g_opt[t] += g_tmp[t]

            loss_history.append(loss_history_o)
            if octave < self.octave_n-1:
                d_intm.append(np.concatenate(d_intm_o, axis=0))

        if self.interp > 1:
            w = np.linspace(0, 1, self.interp+1)
            for t in range(0,self.num_frames-1,self.interp):
                for i in range(1,self.interp):
                    print(t+i, w[i])
                    g_opt[t+i] = g_opt[t]*(1-w[i]) + g_opt[t+self.interp]*w[i]

        # gather outputs
        result = {
            'l': loss_history, 'd_intm': d_intm,
            'v': None, 'c': None}

        # final inference
        p_sty = [None]*self.num_frames
        v_sty = [None]*self.num_frames
        r_sty = [None]*self.num_frames
        d_sty = [None]*self.num_frames
        for t in range(0,self.num_frames,self.batch_size):
            for i in range(self.batch_size):
                feed[self.p[i]] = p[t+i]
                feed[self.opt_ph[i]] = g_opt[t+i]
                if 'd' in self.target_field:
                    feed[self.r[i]] = r[t+i]                

            if self.rotate:
                feed[self.rot_mat] = [np.identity(3)]*self.batch_size

            self.sess.run(self.opt_init, feed)            
            p_, d_, d_img = self.sess.run([self.p_out, self.d_out, self.d_img], feed)

            if 'p' in self.target_field:
                v_ = self.sess.run(self.v, feed)

            for i in range(self.batch_size):
                p_sty[t+i] = p_[i]
                if 'p' in self.target_field:
                    v_sty[t+i] = v_[i]

            d_sty[t:t+self.batch_size] = d_
            r_sty[t:t+self.batch_size] = d_img.astype(np.uint8)

        result['p'] = p_sty
        if 'p' in self.target_field:
            result['v'] = v_sty
        result['d'] = np.array(d_sty)
        result['r'] = np.array(r_sty)

        return result