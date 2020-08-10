#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import numpy as np
import tensorflow as tf
import os
from tqdm import trange
from config import get_config
from util import *
from styler_2p import Styler
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
    styler.load_img(config.resolution)

    params = {}    

    # load particles
    p = []
    r = []
    for i in trange(config.num_frames, desc='load particle'):
        pt_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+i))
        pt = partio.read(pt_path)

        p_id = pt.attributeInfo('id')
        p_pos = pt.attributeInfo('position')
        p_den = pt.attributeInfo('density')

        p_num = pt.numParticles()
        p_ = np.zeros([p_num,2], dtype=np.float32)
        r_ = np.zeros([p_num,1], dtype=np.float32)

        for j in range(p_num):
            p_id_ = pt.get(p_id, j)[0]
            p_[p_id_] = pt.get(p_pos, p_id_)[:-1] # 2d
            r_[p_id_] = pt.get(p_den, p_id_)

        r.append(r_)

        # normalize particle position [0-1]
        px, py = p_[...,0], p_[...,1]
        px /= config.domain[1]
        py /= config.domain[0]
        p_ = np.stack([py,px], axis=-1)
        p.append(p_)

    print('resolution:', config.resolution)
    print('domain:', config.domain)
    print('radius:', config.radius)
    print('normalized px range', px.min(), px.max())
    print('normalized py range', py.min(), py.max())
    print('num particles:', p[0].shape) # the number of particles is fixed

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


    # save density fields
    d_sty = result['d'] # [0-255], uint8
    # d_path = os.path.join(config.log_dir, 'd%03d_%03d.png' % (config.target_frame,config.target_frame+config.num_frames-1))
    # save_image(d_sty, d_path, nrow=5, gray=not 'c' in config.target_field)

    for i, d_sty_ in enumerate(d_sty):
        im = Image.fromarray(d_sty_)
        d_path = os.path.join(config.log_dir, '%03d.png' % (config.target_frame+i))
        im.save(d_path)

    d_intm = result['d_intm']
    for o, d_intm_o in enumerate(d_intm):
        for i, d_intm_ in enumerate(d_intm_o):
            im = Image.fromarray(d_intm_)
            d_path = os.path.join(config.log_dir, 'o%02d_%03d.png' % (o, config.target_frame))
            im.save(d_path)

    # save particles (load using Houdini GPlay)
    c_sty = result['c']
    p_org = []
    for p_ in p:
        # denormalize particle positions
        px, py = p_[...,1], p_[...,0]
        px *= config.domain[1]
        py *= config.domain[0]
        p_org.append(np.stack([px,py], axis=-1))

    for i in range(config.num_frames):
        # create a particle set and attributes
        pt = partio.create()
        position = pt.addAttribute("position",partio.VECTOR,2)
        color = pt.addAttribute("Cd",partio.FLOAT,3)
        radius = pt.addAttribute("radius",partio.FLOAT,1)
        # normal = pt.addAttribute("normal",partio.VECTOR,3)
                
        for pi in range(p_org[i].shape[0]):
            p_ = pt.addParticle()
            pt.set(position, p_, tuple(p_org[i][pi].astype(np.float)))
            pt.set(color, p_, tuple(c_sty[i][pi].astype(np.float)))
            pt.set(radius, p_, (config.radius,))
        
        p_path = os.path.join(config.log_dir, '%03d.bgeo' % (config.target_frame+i))
        partio.write(p_path, pt)

    # visualization using open3d
    bbox = [
        [0,0,-1],
        [config.domain[1],config.domain[0],1], # [X,Y,Z]
        ]
    draw_pt(p_org, pc=c_sty, bbox=bbox, dt=0.1)

def main(config):
    config.dataset = 'dambreak2d'
    config.d_path = 'partio/ParticleData_Fluid_%d.bgeo'

    # from scene
    config.radius = 0.025
    config.support = 4
    config.disc = 2
    config.rest_density = 1000
    config.resolution = [128, 256] # [H,W]
    cell_size = 2*config.radius*config.disc
    config.domain = [float(_*cell_size) for _ in config.resolution] # [H,W]
    config.nsize = max(3-config.disc,1) # 1 is enough if disc is 2, 2 if disc is 1

    # upscaling for rendering
    config.scale = 4
    config.nsize *= config.scale
    config.resolution = [config.resolution[0]*config.scale, config.resolution[1]*config.scale]

    #####################
    # frame range setting
    multi_frame = True
    config.frames_per_opt = 200
    config.window_sigma = 3
    # if multi_frame:
    #     config.target_frame = 1
    #     config.num_frames = 200
    #     config.batch_size = 4
    # else:
    #     config.target_frame = 150
    #     config.num_frames = 1
    #     config.batch_size = 1

    ######
    # color test
    config.target_field = 'c'
    config.lr = 0.01
    config.iter = 100
    config.octave_n = 3
    config.octave_scale = 1.7
    config.clip = False

    # style_list = {
    #     'fire_new': 'fire_new.jpg',
    #     'ben_giles': 'ben_giles.png',
    #     'wave': 'wave.jpeg',
    # }
    # style = 'wave'
    # config.style_target = os.path.join(config.data_dir, 'image', style_list[style])

    config.network = 'vgg_19.ckpt'
    config.w_style = 1
    config.w_content = 0
    config.style_init = 'noise'
    config.style_layer = ['conv2_1','conv3_1']
    config.w_style_layer = [0.5,0.5]
    config.style_mask = True
    config.style_mask_on_ref = False
    config.style_tiling = 2
    config.w_tv = 0.01
    
    if config.w_content == 1:
        config.tag = 'test_%s_%s_%d' % (
            config.target_field, config.content_layer, config.content_channel)
    else:
        style = os.path.splitext(os.path.basename(config.style_target))[0]
        config.tag = 'test_%s_%s' % (
                config.target_field, style)
    
    config.tag += '_%d' % config.num_frames

    run(config)
    
if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)