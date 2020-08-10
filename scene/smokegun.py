#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import argparse
from datetime import datetime
import os
from tqdm import trange
import numpy as np
from PIL import Image
import platform
from subprocess import call
try:
    from manta import *
except ImportError:
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='data/smokegun')
parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%03d.%s')

parser.add_argument("--src_x_pos", type=float, default=0.2)
parser.add_argument("--src_z_pos", type=float, default=0.5)
parser.add_argument("--src_y_pos", type=float, default=0.15)
parser.add_argument("--src_inflow", type=float, default=8)
parser.add_argument("--strength", type=float, default=0.05)
parser.add_argument("--src_radius", type=float, default=0.12)
parser.add_argument("--num_frames", type=int, default=120)
parser.add_argument("--obstacle", type=bool, default=False)

parser.add_argument("--resolution_x", type=int, default=200)
parser.add_argument("--resolution_y", type=int, default=300)
parser.add_argument("--resolution_z", type=int, default=200)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=True)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument("--transmit", type=float, default=0.01)
parser.add_argument("--downup_factor", type=int, default=8)

args = parser.parse_args()

def downup_sample():
    # d_path = os.path.join(args.data_dir, 'd_low')
    # v_path = os.path.join(args.data_dir, 'v_low')
    d_path = os.path.join('E:/lnst/data/smokegun', 'd_low')
    v_path = os.path.join('E:/lnst/data/smokegun', 'v_low')
    for f_path in [d_path, v_path]:
        if not os.path.exists(f_path):
            os.mkdir(f_path)

    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    org_res = [res_z,res_y,res_x]
    down_res = [r//args.downup_factor for r in org_res]
    d_ = np.zeros(org_res, dtype=np.float32)
    
    # solver params
    gs = vec3(res_x, res_y, res_z)
    buoyancy = vec3(0,args.buoyancy,0)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)
    
    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
    if args.open_bound:
        setOpenBound(flags, args.bWidth,'xXyYzZ', FlagOutflow|FlagEmpty)

    radius = gs.x*args.src_radius
    center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
    source = s.create(Sphere, center=center, radius=radius)

    if args.obstacle:
        obs_radius = gs.x*0.15
        obs_center = gs*vec3(0.7, 0.5, 0.5)
        obs = s.create(Sphere, center=obs_center, radius=obs_radius)
        obs.applyToGrid(grid=flags, value=FlagObstacle)
    
    if (GUI):
        gui = Gui()
        gui.show(True)
        #gui.pause()

    def resize(v, vshape, order=3):
        import skimage.transform
        v0 = skimage.transform.resize(v[...,0], vshape, order=3).astype(np.float32)
        v1 = skimage.transform.resize(v[...,1], vshape, order=3).astype(np.float32)
        v2 = skimage.transform.resize(v[...,2], vshape, order=3).astype(np.float32)
        return np.stack([v0,v1,v2], axis=-1).astype(np.float32)

    for t in trange(args.num_frames, desc='downup_sample'):
        source.applyToGrid(grid=density, value=1)

        # save density
        copyGridToArrayReal(density, d_)
        d_file_path = os.path.join(d_path, args.path_format % (t, 'npz'))
        np.savez_compressed(d_file_path, x=d_)

        d_file_path = os.path.join(d_path, args.path_format % (t, 'png'))
        transmit = np.exp(-np.cumsum(d_[::-1], axis=0)*args.transmit)
        d_img = np.sum(d_*transmit, axis=0)
        d_img /= d_img.max()
        im = Image.fromarray((d_img[::-1]*255).astype(np.uint8))
        im.save(d_file_path)

        v_file_path = os.path.join(args.data_dir, 'v', args.path_format % (t, 'npz'))
        with np.load(v_file_path) as data:
            v_ = data['x']
        v_ = resize(resize(v_, down_res), org_res)

        # save velocity
        v_file_path = os.path.join(v_path, args.path_format % (t, 'npz'))
        np.savez_compressed(v_file_path, x=v_)

        # advect density
        copyArrayToGridMAC(v_, vel)
        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
                            openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        s.step()

def main():
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    field_type = ['d', 'v']
    for field in field_type:
        field_path = os.path.join(args.data_dir,field)
        if not os.path.exists(field_path):
            os.mkdir(field_path)

    args_file = os.path.join(args.data_dir, 'args.txt')
    with open(args_file, 'w') as f:
        print('%s: arguments' % datetime.now())
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    v_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
    
    # solver params
    gs = vec3(res_x, res_y, res_z)
    buoyancy = vec3(0,args.buoyancy,0)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)
    pressure = s.create(RealGrid)

    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
    if args.open_bound:
        setOpenBound(flags, args.bWidth,'xXyYzZ', FlagOutflow|FlagEmpty)

    radius = gs.x*args.src_radius
    center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
    source = s.create(Sphere, center=center, radius=radius)

    if args.obstacle:
        obs_radius = gs.x*0.15
        obs_center = gs*vec3(0.7, 0.5, 0.5)
        obs = s.create(Sphere, center=obs_center, radius=obs_radius)
        obs.applyToGrid(grid=flags, value=FlagObstacle)
    
    if (GUI):
        gui = Gui()
        gui.show(True)
        #gui.pause()

    for t in trange(args.num_frames, desc='sim'):
        source.applyToGrid(grid=density, value=1)
        source.applyToGrid(grid=vel, value=vec3(args.src_inflow,0,0))

        # save density
        copyGridToArrayReal(density, d_)
        d_file_path = os.path.join(args.data_dir, 'd', args.path_format % (t, 'npz'))
        np.savez_compressed(d_file_path, x=d_)

        d_file_path = os.path.join(args.data_dir, 'd', args.path_format % (t, 'png'))
        transmit = np.exp(-np.cumsum(d_[::-1], axis=0)*args.transmit)
        d_img = np.sum(d_*transmit, axis=0)
        d_img /= d_img.max()
        im = Image.fromarray((d_img[::-1]*255).astype(np.uint8))
        im.save(d_file_path)

        # save velocity
        v_file_path = os.path.join(args.data_dir, 'v', args.path_format % (t, 'npz'))
        copyGridToArrayMAC(vel, v_)
        np.savez_compressed(v_file_path, x=v_)

        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
                            openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=args.adv_order,
                            openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        vorticityConfinement(vel=vel, flags=flags, strength=args.strength)

        setWallBcs(flags=flags, vel=vel)
        addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
        solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
        setWallBcs(flags=flags, vel=vel)

        s.step()

if __name__ == '__main__':
    main()
    downup_sample()