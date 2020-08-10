#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import trange
import struct
from config import get_config
from util import *
from transform import g2p, p2g, p2g_wavg
import sys
sys.path.append('E:/partio/build/py/Release')
import partio

class SimG2P(object):
    def __init__(self, self_dict):
        # get arguments
        for arg in vars(self_dict):
            setattr(self, arg, getattr(self_dict,arg))
        
        self.sess = tf.compat.v1.InteractiveSession()

        # particle position at t
        x_shp = [None,3]
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=x_shp, name='x')
        x = tf.expand_dims(self.x, axis=0) # [1,None,3]

        # velocity field to sample
        u_shp = [None,None,None,3]
        self.u = tf.compat.v1.placeholder(dtype=tf.float32, shape=u_shp, name='u')
        u = tf.expand_dims(self.u, axis=0) # [1,None,None,None,3]
       
        # grid to particle velocity
        v = g2p(u, x, is_2d=False)

        ####
        # RK4 velocity sampling
        x1 = x + v*0.5
        v1 = g2p(u, x1, is_2d=False)

        x2 = x + v1*0.5
        v2 = g2p(u, x2, is_2d=False)

        x3 = x + v2
        v3 = g2p(u, x3, is_2d=False)
        v = (v + v1*2 + v2*2 + v3)/6
        ####

        # advect to t+1
        time_step = 0.5
        x_adv = x + v*time_step
        self.x_adv = x_adv[0]

        ############
        # particle position displacement for optimization
        self.v = tf.compat.v1.placeholder(dtype=tf.float32, shape=x_shp, name='v')
        self.optv = tf.Variable(self.v, validate_shape=False, name='v_opt')
        v_opt = tf.reshape(self.optv, tf.shape(self.v))
        self.pv = v_opt
        pv = tf.expand_dims(v_opt, axis=0)
        self.x_hat = x + pv

        # density splatting
        self.res = tf.compat.v1.placeholder(tf.int32, [3], name='resolution')
        d_rec = p2g(self.x_hat, self.domain, self.res, self.radius, self.rest_density, self.nsize, kernel='cubic', support=4, clip=False, is_2d=False)
        pressure = d_rec - self.rest_density
        pressure = tf.where(d_rec>0, pressure, tf.zeros_like(pressure))

        # L2 Loss: pressure
        self.pres_loss = tf.reduce_mean(pressure**2)
        self.loss = self.pres_loss # + 0.1*tf.reduce_mean(tf.compat.v1.image.total_variation(pressure[0,...,0])) # weak TV loss

        self.opt_init = tf.compat.v1.initializers.variables([self.optv])
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.optv])

        ############
        # multi-scale density sampling

        # ground truth density field at t+1
        d_shp = [None,None,None]
        self.d = tf.compat.v1.placeholder(dtype=tf.float32, shape=d_shp, name='d')
        d = tf.expand_dims(tf.expand_dims(self.d, axis=0), axis=-1)
        # d = resize_tf(d, self.res, method=tf.image.ResizeMethod.BILINEAR, is_3d=True)

        # particle density sampling at t+1
        r = []
        for o in range(self.octave_n):
            if o > 0:
                d_hi = d_hat
                d_ = d - d_hi[:,:,::-1]
            else:
                d_ = d
            r_ = g2p(d_, self.x_hat, is_2d=False)
            r.append(r_)

            factor = self.octave_scale**o
            d_hat = p2g_wavg(self.x_hat, r_, self.domain, self.res, self.radius, self.nsize, kernel='cubic', is_2d=False, clip=False, support=self.support/factor)
            if o > 0:
                d_hat += d_hi
        
        self.r_smp = tf.concat(r, axis=-1)[0]
        self.d_smp = tf.clip_by_value(d_hat[0,...,0], 0, 1)
        self.d_diff = (d[:,:,::-1] - d_hat)[0,:,::-1,:,0]

        # simple advection test
        r_shp = [None,1]
        self.r = tf.compat.v1.placeholder(dtype=tf.float32, shape=r_shp, name='r')
        r_ = tf.expand_dims(self.r, axis=0) # [1,None,3]
        d_rec = p2g_wavg(x_adv, r_, self.domain, self.res, self.radius, self.nsize, kernel='cubic', is_2d=False, clip=False, support=4)
        self.d_rec = d_rec[0,...,0]

    def sample(self, d, disc=1, threshold=0, p0=None, p_id=None):
        '''        
        sample particles where d's value is higher than threshold
        '''
        # pid = np.where(d > threshold)
        # add pt only in src region
        pid = np.where(d[76:124,231:279,16:64] > threshold)
        pid = np.array(pid).transpose([1,0]).astype(np.float)
        pid += np.array([76,231,16])
        
        cell_size = 1/disc
        offset = cell_size/2
        p = []
        for i in range(disc):
            for j in range(disc):
                for k in range(disc):
                    p_ = pid + offset + np.array([cell_size*i, cell_size*j, cell_size*k]) 
                    p.append(p_)
        p = np.concatenate(p, axis=0)

        # normalize to [0,1]
        pz, py, px = p[:,0], p[:,1], p[:,2]
        pz /= d.shape[0]
        py /= d.shape[1]
        px /= d.shape[2]
        p = np.stack([pz,py,px], axis=-1)

        # if there are new particles, add to prev
        if len(p) > 0:
            if p_id is None:
                p_id = np.arange(p.shape[0])
            else:
                p_id0 = p_id[-1]+1
                p_id_new = np.arange(p_id0, p_id0+p.shape[0])
                p_id = np.concatenate([p_id, p_id_new])

            if p0 is not None:
                p = np.concatenate([p0, p], axis=0)

        return p, p_id

    def naive_adv(self, p, u, r):
        '''
        reconstruct density field from p_t' with r
        '''
        # advect particle to t+1 first
        feed = {self.res: self.resolution}
        feed[self.x] = p
        feed[self.u] = u
        feed[self.r] = r
        p_adv, d_rec = self.sess.run([self.x_adv, self.d_rec], feed)
        return p_adv, d_rec

    def optimize(self, p, p_id, d, u):
        '''
        1. advect p_t using u_t then optimize for redistribution
        2. sample new particles where particles don't cover (src region)
        3. sample particle density from d_(t+1)
        '''
        # advect particle to t+1 first
        feed = {self.res: self.resolution}
        feed[self.x] = p # p_t
        feed[self.u] = u
        p = self.sess.run(self.x_adv, feed)

        # optimize for particle redistribution
        feed[self.x] = p # p_t'
        feed[self.v] = np.zeros_like(p)

        # init variables
        self.sess.run(self.opt_init, feed)
        self.sess.run(tf.compat.v1.variables_initializer(self.opt.variables()), feed)

        # optimize particle positions
        l = []
        for _ in range(self.iter):
            # self.sess.run(self.train_op, feed)
            l_, _ = self.sess.run([self.loss, self.train_op], feed)
            l.append(l_)

        # seed particles
        feed[self.d] = d # d_t
        d_diff = self.sess.run(self.d_diff, feed)
        p_new = self.sess.run(self.x_hat, feed)[0]
        p, p_id = self.sample(d_diff, disc=self.disc, threshold=self.threshold, p0=p_new, p_id=p_id)
        
        # sample density at new position
        feed[self.x_hat] = p[None,:] # p_t'
        p_den = self.sess.run(self.r_smp, feed)

        result = {
            'p': p,
            'p_id': p_id,
            'p_den': p_den,
            'l': l,
        }

        # for debug
        result['d_diff'] = np.mean(d_diff, axis=0)
        d_smp = self.sess.run(self.d_smp, feed)
        result['d_smp'] = d_smp

        return result

def run(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id # "0, 1" for multiple

    prepare_dirs_and_logger(config)
    tf.compat.v1.set_random_seed(config.seed)
    config.rng = np.random.RandomState(config.seed)

    resampler = SimG2P(config)

    # load input density fields
    for t in trange(config.num_frames, desc='load density'): # last one for mask
        d_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+t))
        with np.load(d_path) as data:
            d = data['x'][:,::-1] # [D,H,W], [0-1]
    
        # mantaflow dataset
        v_path = os.path.join(config.data_dir, config.dataset, config.v_path % (config.target_frame+t))
        with np.load(v_path) as data:
            v_ = data['x'] # [D,H,W,3]
            vx = np.dstack((v_,np.zeros((v_.shape[0],v_.shape[1],1,v_.shape[3]))))
            vx = (vx[:,:,1:,0] + vx[:,:,:-1,0]) * 0.5
            vy = np.hstack((v_,np.zeros((v_.shape[0],1,v_.shape[2],v_.shape[3]))))
            vy = (vy[:,1:,:,1] + vy[:,:-1,:,1]) * 0.5
            vz = np.vstack((v_,np.zeros((1,v_.shape[1],v_.shape[2],v_.shape[3]))))
            vz = (vz[1:,:,:,2] + vz[:-1,:,:,2]) * 0.5
            v_ = np.stack([vx,vy,vz], axis=-1)
            v_ = v_[:,::-1]
        
        vx = v_[...,0] / v_.shape[2] * config.scale
        vy = -v_[...,1] / v_.shape[1] * config.scale
        vz = v_[...,2] / v_.shape[0] * config.scale
        u = np.stack([vz,vy,vx], axis=-1)

        if config.resampling: 
            if t == 0:
                n_prev = 0

                # sampling at the beginning wo opt.
                p, p_id = resampler.sample(d, disc=config.disc, threshold=0)
            
            result = resampler.optimize(p, p_id, d, u)
        
            p = result['p']
            p_id = result['p_id']
            p_den = result['p_den']
            # d_diff = result['d_diff']
            # plt.imshow(d_diff); plt.show()
            l = result['l'][-1] # last loss
            d_smp = result['d_smp']
        else:
            if t == 0:
                n_prev = 0

                # sampling at the beginning wo opt.
                p, p_id = resampler.sample(d, disc=config.disc, threshold=0)
                p_src = p
            else:
                # simply source particles of t=0
                p = np.concatenate([p,p_src], axis=0)
                p_id = np.arange(p.shape[0])
            
            p_den = np.ones([p.shape[0],1])
            p, d_smp = resampler.naive_adv(p, u, p_den)
            l = 0

        print(t, 'num particles', p.shape[0], '(+%d)' % (p.shape[0]-n_prev), 'loss', l)
        n_prev = p.shape[0]

        # convert to the original domain coordinate
        px, py, pz = p[...,2], 1-p[...,1], p[...,0]
        p_ = np.stack([
            px*config.domain[2],
            py*config.domain[1],
            pz*config.domain[0]], axis=-1)

        # create a particle set and attributes
        pt = partio.create()
        pid = pt.addAttribute('id',partio.INT,1)
        position = pt.addAttribute("position",partio.VECTOR,3)
        if p_den.shape[1] > 1:
            density = pt.addAttribute('density',partio.VECTOR,p_den.shape[1])
        else:
            density = pt.addAttribute('density',partio.FLOAT,1)
        color = pt.addAttribute("Cd",partio.FLOAT,3)
        radius = pt.addAttribute("radius",partio.FLOAT,1)
        
        for i in range(p_.shape[0]):
            pt_ = pt.addParticle()
            pt.set(pid, pt_, (int(p_id[i]),))
            pt.set(position, pt_, tuple(p_[i].astype(np.float)))
            if p_den.shape[1] > 1:
                pt.set(density, pt_, tuple(p_den[i].astype(np.float)))
            else:
                pt.set(density, pt_, (float(p_den[i]),))
            pt.set(color, pt_, tuple(np.array([p_den[i,0]]*3,dtype=np.float)))
            pt.set(radius, pt_, (config.radius,))
        
        # save particle
        p_path = os.path.join(config.log_dir, '%03d.bgeo' % (config.target_frame+t))
        partio.write(p_path, pt)

        # save density image
        transmit = np.exp(-np.cumsum(d_smp[::-1], axis=0)*config.transmit)
        d_img = np.sum(d_smp*transmit, axis=0)
        d_img /= d_img.max()
        im = Image.fromarray((d_img[::-1]*255).astype(np.uint8))
        im_path = os.path.join(config.log_dir, '%03d.png' % (config.target_frame+t))
        im.save(im_path)

    stat_path = os.path.join(config.log_dir, 'stat.txt')
    with open(stat_path, 'w') as f:
        f.write('num particles %d\n' % p.shape[0])
        f.write('loss %.2f' % l)

    # # visualize last frame
    # bbox = [
    #     [0,0,0],
    #     [config.domain[2],config.domain[1],config.domain[0]],
    #     ]
    # if config.octave_n == 1:
    #     pc = np.concatenate([p_den]*3, axis=-1)
    # else:
    #     pc = np.concatenate([p_den[:,0,None]]*3, axis=-1)
    # draw_pt([p_], pc=[pc], bbox=bbox, is_2d=False)

def main(config):
    config.dataset = 'smokegun'
    config.d_path = 'd_low/%03d.npz'
    config.v_path = 'v_low/%03d.npz'

    config.num_frames = 120
    config.target_frame = 0 # 120 - config.num_frames

    # config.target_frame = 60
    # config.num_frames = 3
    
    config.scale = 1
    config.domain = [_*config.scale for _ in [200,300,200]]
    config.resolution = [int(_) for _ in config.domain]

    config.disc = 1
    cell_size = 1 # == 2*radius*disc
    config.radius = cell_size/config.disc/2
    config.nsize = 1
    config.support = 4
    config.rest_density = 1000
    config.threshold = 0.01
    config.lr = 0.0005
    config.iter = 20
    config.transmit = 0.01
    config.octave_n = 2
    if config.octave_n > 1:
        config.octave_scale = 2
    else:
        config.octave_scale = 1

    # resampling or naive advection
    config.resampling = True
    if config.resampling:
        config.tag = 'n%d_it%d_o%d' % (config.num_frames, config.iter, config.octave_n)
    else:
        config.tag = 'naive_n%d' % config.num_frames

    run(config)
    
if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
