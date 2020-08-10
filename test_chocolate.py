#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import numpy as np
import tensorflow as tf
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

    # load particles
    nmin, nmax = np.iinfo(np.int32).max, 0
    for i in range(config.num_frames):
        pt_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+i))
        pt = partio.read(pt_path)
        p_num = pt.numParticles()
        nmin = min(p_num, nmin)
        nmax = max(p_num, nmax)
        
    print('# range:', nmin, nmax)

    p = []
    # r = []
    for i in trange(config.num_frames, desc='load particle'): # last one for mask
        pt_path = os.path.join(config.data_dir, config.dataset, config.d_path % (config.target_frame+i))
        pt = partio.read(pt_path)

        p_attr_id = pt.attributeInfo('id')
        p_attr_pos = pt.attributeInfo('position')
        # p_attr_den = pt.attributeInfo('density')

        p_ = np.ones([nmax,3], dtype=np.float32)*-1
        # r_ = np.zeros([nmax,1], dtype=np.float32)

        p_num = pt.numParticles()
        for j in range(p_num):
            p_id_j = pt.get(p_attr_id, j)[0]
            p_[p_id_j] = pt.get(p_attr_pos, p_id_j)
            # r_[p_id_j] = pt.get(p_attr_den, p_id_j)
        # r.append(r_)

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

    params['p'] = p

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

    # save particle (load using Houdini GPlay)
    p_sty = result['p']
    p = []
    # v_sty = result['v']
    # v = []
    for i in range(config.num_frames):
        # denormalize particle positions
        px, py, pz = p_sty[i][...,2], p_sty[i][...,1], p_sty[i][...,0]
        px *= config.domain[2]
        py *= config.domain[1]
        pz *= config.domain[0]
        p_sty_ = np.stack([px,py,pz], axis=-1)
        p.append(p_sty_)

        # # denormalize particle displacement for stylization
        # vx, vy, vz = v_sty[i][...,2], v_sty[i][...,1], v_sty[i][...,0]
        # vx *= config.domain[2]
        # vy *= config.domain[1]
        # vz *= config.domain[0]
        # v_sty_ = np.stack([vx,vy,vz], axis=-1)
        # v.append(v_sty_)

        # create a particle set and attributes
        pt = partio.create()
        position = pt.addAttribute("position",partio.VECTOR,3)
        # color = pt.addAttribute("Cd",partio.FLOAT,3)
        radius = pt.addAttribute("radius",partio.FLOAT,1)
        # normal = pt.addAttribute("normal",partio.VECTOR,3)
        
        for p_sty_i in p_sty_:
            if p_sty_i[0] < 0: continue
            p_ = pt.addParticle()
            pt.set(position, p_, tuple(p_sty_i.astype(np.float)))
            pt.set(radius, p_, (config.radius,))
        
        p_path = os.path.join(config.log_dir, '%03d.bgeo' % (config.target_frame+i))
        partio.write(p_path, pt)

    r_sty = result['r']
    for i, r_sty_ in enumerate(r_sty):
        im = Image.fromarray(r_sty_)
        d_path = os.path.join(config.log_dir, '%03d.png' % (config.target_frame+i))
        im.save(d_path)

    d_intm = result['d_intm']
    for o, d_intm_o in enumerate(d_intm):
        for i, d_intm_ in enumerate(d_intm_o):
            if d_intm_ is None: continue
            im = Image.fromarray(d_intm_)
            d_path = os.path.join(config.log_dir, 'o%02d_%03d.png' % (o, config.target_frame+i))
            im.save(d_path)

    # visualization using open3d
    bbox = [
        [0,0,0],
        [config.domain[2],config.domain[1],config.domain[0]], # [X,Y,Z]
        ]
    draw_pt(p, bbox=bbox, dt=0.1, is_2d=False) # pv=v, 

def main(config):
    config.dataset = 'chocolate'
    config.d_path = 'partio/ParticleData_Fluid_%d.bgeo'

    # from scene
    config.radius = 0.025
    config.support = 4
    config.disc = 2 # 1 or 2
    config.rest_density = 1000
    config.resolution = [128,128,128] # original resolution, # [D,H,W]
    cell_size = 2*config.radius*config.disc
    config.domain = [float(_*cell_size) for _ in config.resolution] # [D,H,W]
    config.nsize = max(3-config.disc,1) # 1 is enough if disc is 2, 2 if disc is 1
    
    # upscaling for rendering
    config.resolution = [200,200,200]

    # default settings
    config.lr = 0.002
    config.iter = 20
    config.resize_scale = 1
    config.transmit = 0.2 # 0.01, 1
    config.clip = False # ignore particles outside of domain
    config.num_kernels = 1
    config.k = 3
    config.network = 'tensorflow_inception_graph.pb'

    config.octave_n = 2
    config.octave_scale = 1.8
    config.render_liquid = True
    config.rotate = False
    config.style_layer = ['conv2d2','mixed3b','mixed4b']
    config.w_style_layer = [1,1,1]
    
    #####################
    # frame range setting
    config.frames_per_opt = 120
    config.batch_size = 1
    config.window_sigma = 9
    # multi_frame = True
    # if multi_frame:
    #     config.target_frame = 1
    #     config.num_frames = 120
    #     config.interp = 1

    #     ######
    #     # interpolation test
    #     interpolate = False
    #     if interpolate:
    #         config.interp = 5
    #         n = (config.num_frames-1)//config.interp
    #         config.num_frames = n*config.interp + 1
    #         assert (config.num_frames - 1) % config.interp == 0
    #     #####
    #     
    # else:
    #     config.target_frame = 90
    #     config.num_frames = 1
    #     config.interp = 1
        
    ######
    # position test
    config.target_field = 'p'
    semantic = False
    pressure = False

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

    # pressure test
    if pressure:
        config.w_pressure = 1e12 # 1e10 ~ 1e12
    #####

    if config.w_content == 1:
        config.tag = 'test_%s_%s_%d' % (
            config.target_field, config.content_layer, config.content_channel)
    else:
        style = os.path.splitext(os.path.basename(config.style_target))[0]
        config.tag = 'test_%s_%s' % (
                config.target_field, style)

    config.tag += '_%d_intp%d' % (config.num_frames, config.interp)
   
    quick_test = False
    if quick_test:
        config.scale = 1
        config.iter = 0
        config.octave_n = 1
    
    run(config)
    
if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)