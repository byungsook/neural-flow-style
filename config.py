#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import argparse
from util import str2bool

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# path
path_arg = add_argument_group('Path')
path_arg.add_argument("--data_dir", type=str, default='data')
path_arg.add_argument("--log_dir", type=str, default='log')
path_arg.add_argument("--model_dir", type=str, default='model')
path_arg.add_argument("--d_path", type=str, default='d/%03d.npz')
path_arg.add_argument("--v_path", type=str, default='v/%03d.npz')
path_arg.add_argument("--tag", type=str, default='test')

# dataset
data_arg = add_argument_group('Data')
data_arg.add_argument("--dataset", type=str, default='smokegun')
data_arg.add_argument("--target_frame", type=int, default=70)
data_arg.add_argument("--num_frames", type=int, default=1)
data_arg.add_argument("--scale", type=float, default=2.0)

# network
network_arg = add_argument_group('Network')
network_arg.add_argument("--network", type=str, default='tensorflow_inception_graph.pb',
    choices=['tensorflow_inception_graph.pb','vgg_19.ckpt'])
network_arg.add_argument("--pool1", type=str2bool, default=False)
network_arg.add_argument("--batch_size", type=int, default=1)

# grid
grid_arg = add_argument_group('Grid')
grid_arg.add_argument("--resolution", nargs='+', type=int, default=[384,288]) # HW or DHW
grid_arg.add_argument("--adv_order", type=int, default=1, choices=[1,2], help='SL or MacCormack')

# particle
pt_arg = add_argument_group('Particle')
pt_arg.add_argument("--domain", nargs='+', type=int, default=[12.8,12.8,12.8]) # HW or DHW
pt_arg.add_argument("--radius", type=float, default=0.025, help='kernel radius for density estimation')
pt_arg.add_argument("--disc", type=int, default=2, help='grid discretization')
pt_arg.add_argument("--nsize", type=int, default=1, help='# neighbors cells to check')
pt_arg.add_argument("--rest_density", type=float, default=1000)
pt_arg.add_argument("--w_pressure", type=float, default=0)
pt_arg.add_argument("--w_density", type=float, default=0)
pt_arg.add_argument("--window_sigma", type=float, default=2)
pt_arg.add_argument("--interp", type=int, default=1)
pt_arg.add_argument("--support", type=float, default=4)
pt_arg.add_argument("--k", type=int, default=3)
pt_arg.add_argument("--clip", type=str2bool, default=False, help='whether to clamp particle pos to domain or not')

# rendering
render_arg = add_argument_group('Render')
render_arg.add_argument("--resize_scale", type=float, default=1.0, help='to upscale rendering')
render_arg.add_argument("--transmit", type=float, default=0.01)
render_arg.add_argument("--rotate", type=str2bool, default=False)
render_arg.add_argument('--phi0', type=int, default=-5) # latitude (elevation) start
render_arg.add_argument('--phi1', type=int, default=5) # latitude end
render_arg.add_argument('--phi_unit', type=int, default=5)
render_arg.add_argument('--theta0', type=int, default=-10) # longitude start
render_arg.add_argument('--theta1', type=int, default=10) # longitude end
render_arg.add_argument('--theta_unit', type=int, default=10)
render_arg.add_argument('--v_batch', type=int, default=1, help='# of rotation matrix for batch process')
render_arg.add_argument('--n_views', type=int, default=9, help='# of view points')
render_arg.add_argument('--sample_type', type=str, default='poisson',
                        choices=['uniform', 'poisson', 'both'])
render_arg.add_argument("--render_liquid", type=str2bool, default=False)

# optimizer
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument("--target_field", type=str, default='p', choices=['d', 'p', 'c'])
opt_arg.add_argument("--optimizer", type=str, default='adam')
opt_arg.add_argument("--iter", type=int, default=20)
opt_arg.add_argument("--lr", type=float, default=0.0007)
opt_arg.add_argument("--lr_scale", type=float, default=1)
opt_arg.add_argument("--octave_n", type=int, default=2)
opt_arg.add_argument("--octave_scale", type=float, default=1.8)
opt_arg.add_argument("--frames_per_opt", type=int, default=10)

# style
style_arg = add_argument_group('Style')
style_arg.add_argument("--content_layer", type=str, default='mixed4d_3x3_bottleneck_pre_relu')
style_arg.add_argument("--content_channel", type=int, default=139)
style_arg.add_argument("--w_content", type=float, default=1)
style_arg.add_argument("--w_content_amp", type=float, default=100)
style_arg.add_argument("--content_target", type=str, default='')
style_arg.add_argument("--top_k", type=int, default=5)
style_arg.add_argument("--style_layer", nargs='+', type=str, default=['conv3_1']) #['conv2d2','mixed3a','mixed4a','mixed5a'])
style_arg.add_argument("--w_style", type=float, default=0)
style_arg.add_argument("--w_style_layer", nargs='+', type=float, default=[1]) #[1,0.01,0.3,10])
style_arg.add_argument("--hist_layer", nargs='+', type=str, default=['input'])
style_arg.add_argument("--w_hist", type=float, default=0)
style_arg.add_argument("--w_hist_layer", nargs='+', type=float, default=[1])
style_arg.add_argument("--w_tv", type=float, default=0)
style_arg.add_argument("--style_target", type=str, default='') # data/image/fire_new.jpg
style_arg.add_argument("--style_mask", type=str2bool, default=False)
style_arg.add_argument("--style_mask_on_ref", type=str2bool, default=False)
style_arg.add_argument("--style_tiling", type=int, default=1)
style_arg.add_argument("--style_init", type=str, default='noise', choices=['noise','style'])

# misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument("--seed", type=int, default=123)
misc_arg.add_argument('--gpu_id', type=str, default='0', help='-1:cpu')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed