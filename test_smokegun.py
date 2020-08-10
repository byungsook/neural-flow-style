#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import trange
from config import get_config
from util import *
from styler_3p import Styler
import sys
sys.path.append('E:/partio/build/py/Release')
import partio

def run(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id # "0, 1" for multiple

    prepare_dirs_and_logger(config)
    tf.compat.v1.set_random_seed(config.seed)
    config.rng = np.random.RandomState(config.seed)

    styler = Styler(config)
    styler.load_img(config.resolution[1:])

    params = {}

    # the number of particles range
    nmin, nmax = np.iinfo(np.int32).max, 0
    for i in range(config.num_frames):
        pt_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+i))
        pt = partio.read(pt_path)
        p_num = pt.numParticles()
        nmin, nmax = min(nmin,p_num), max(nmax,p_num)

    print('# range:', nmin, nmax)
    
    p, r = [], []
    for i in trange(config.num_frames, desc='load particle'):
        pt_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+i))
        pt = partio.read(pt_path)
    
        p_id = pt.attributeInfo('id')
        p_pos = pt.attributeInfo('position')
        p_den = pt.attributeInfo('density')

        p_ = np.ones([nmax,3], dtype=np.float32)*-1
        r_ = np.zeros([nmax,config.num_kernels], dtype=np.float32)

        p_num = pt.numParticles()
        for j in range(p_num):
            p_id_j = pt.get(p_id, j)[0]
            p_[j] = pt.get(p_pos, p_id_j)
            r_[j] = pt.get(p_den, p_id_j)

        r.append(r_)

        # normalize particle position [0-1]
        px, py, pz = p_[...,0], p_[...,1], p_[...,2]
        px /= config.domain[2]
        py /= config.domain[1]
        pz /= config.domain[0]
        p_ = np.stack([pz,py,px], axis=-1)
        p.append(p_)

    
    print('resolution:', config.resolution)
    print('domain:', config.domain)
    print('radius:', config.radius)
    print('normalized px range', px.min(), px.max())
    print('normalized py range', py.min(), py.max())
    print('normalized pz range', pz.min(), pz.max())

    params['p'] = p
    params['r'] = r
        
    # styler.render_test(params)
    result = styler.run(params)

    # save loss plot
    l = result['l']
    lb = []
    for o, l_ in enumerate(l):
        lb_, = plt.plot(range(len(l_)), l_, label='oct %d' % o)
        lb.append(lb_)
    plt.legend(handles=lb)
    # plt.show()
    plot_path = os.path.join(config.log_dir, 'loss_plot.png')
    plt.savefig(plot_path)

    r_sty = result['r']
    for i, r_sty_ in enumerate(r_sty):
        im = Image.fromarray(r_sty_)
        d_path = os.path.join(config.log_dir, '%03d.png' % (config.target_frame+i))
        im.save(d_path)

    d_sty = result['d']
    for i, d_sty_ in enumerate(d_sty):
        d_path = os.path.join(config.log_dir, '%03d.npz' % (config.target_frame+i))
        np.savez_compressed(d_path, x=d_sty_[:,::-1])

    d_intm = result['d_intm']
    for o, d_intm_o in enumerate(d_intm):
        for i, d_intm_ in enumerate(d_intm_o):
            if d_intm_ is None: continue
            im = Image.fromarray(d_intm_)
            d_path = os.path.join(config.log_dir, 'o%02d_%03d.png' % (o, config.target_frame+i))
            im.save(d_path)
        
def main(config):
    config.dataset = 'smokegun'
    
    # config.d_path = 'pt_low_o1/%03d.npz'
    # config.num_kernels = 1

    config.d_path = 'pt_low_o2/%03d.bgeo'
    config.num_kernels = 2

    config.kernel_scale = 2
    config.support = 4

    config.disc = 1
    cell_size = 1 # == 2*radius*disc
    config.radius = cell_size/config.disc/2
    config.nsize = 1
    config.rest_density = 1000
    config.resolution = [200,300,200]
    config.domain = [200,300,200]
    config.clip = False
    config.w_density = 0
    config.k = 3

    config.window_sigma = 3
    config.batch_size = 1
    config.frames_per_opt = 1

    config.target_field = 'd'
    config.lr = 0.1
    config.network = 'tensorflow_inception_graph.pb'
    config.style_layer = ['conv2d2','mixed3b','mixed4b']
    config.w_style_layer = [1,1,1]
    config.octave_n = 1
    config.octave_scale = 1.8
    config.transmit = 0.01 # 0.01, 5
    config.iter = 20
    config.resize_scale = 300/config.resolution[0]
    config.rotate = False

    multi_frame = False
    config.interp = 1
    config.batch_size = 1
    config.frames_per_opt = 1
    # if multi_frame:
    #     config.num_frames = 120
    #     config.target_frame = 0
    # else:
    #     config.target_frame = 70
    #     config.num_frames = 1

    semantic = True
    density_reg = False

    # if semantic:
    #     config.w_style = 0
    #     config.w_content = 1
    #     config.content_layer = 'mixed3b_3x3_bottleneck_pre_relu'
    #     config.content_channel = 44 # net
    # else:
    #     # style
    #     config.w_style = 1
    #     config.w_content = 0

    #     style_list = {
    #         'spiral': 'pattern1.png',
    #         'fire_new': 'fire_new.jpg',
    #         'ben_giles': 'ben_giles.png',
    #         'wave': 'wave.jpeg',
    #     }
    #     style = 'spiral'
    #     config.style_target = os.path.join(config.data_dir, 'image', style_list[style])

    # density regularization
    if density_reg:
        config.w_density = 1e-6

    # if config.w_content == 1:
    #     config.tag = 'test_%s_%s_%d' % (
    #         config.target_field, config.content_layer, config.content_channel)
    # else:
    #     style = os.path.splitext(os.path.basename(config.style_target))[0]
    #     config.tag = 'test_%s_%s' % (
    #             config.target_field, style)

    # config.tag += '_%d' % config.num_frames

    run(config)
    
if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)