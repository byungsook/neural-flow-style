# https://github.com/Ryo-Ito/spatial_transformer_network

import tensorflow as tf
import numpy as np
from poisson import PoissonDisc

def mgrid(*args, **kwargs):
    """
    create orthogonal grid
    similar to np.mgrid

    Parameters
    ----------
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grid : tf.Tensor [len(args), args[0], ...]
        orthogonal grid
    """
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)
    low = tf.to_float(low)
    high = tf.to_float(high)
    coords = (tf.linspace(low, high, arg) for arg in args)
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid


def batch_mgrid(n_batch, *args, **kwargs):
    """
    create batch of orthogonal grids
    similar to np.mgrid

    Parameters
    ----------
    n_batch : int
        number of grids to create
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grids : tf.Tensor [n_batch, len(args), args[0], ...]
        batch of orthogonal grids
    """
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    return grids

def batch_warp2d(imgs, mappings, sample_shape):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, xlen, ylen, 2]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    # n_batch = tf.shape(imgs)[0]
    n_batch = sample_shape[0]
    coords = tf.reshape(mappings, [n_batch, 2, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])

    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat, sample_shape)
    return output

def batch_warp3d(imgs, mappings, sample_shape):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, 3, xlen, ylen, zlen]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = sample_shape[0]
    coords = tf.reshape(mappings, [n_batch, 3, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    z_coords_flat = tf.reshape(z_coords, [-1])

    output = _interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat, sample_shape)
    return output

def _repeat(base_indices, n_repeats):
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    #     tf.reshape(tf.to_float(base_indices), [-1, 1]),
    #     tf.ones([1, n_repeats], dtype=tf.float32))
    # base_indices = tf.to_int32(base_indices)
    return tf.reshape(base_indices, [-1])

def _interpolate2d(imgs, x, y, sample_shape):
    # n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    n_channel = tf.shape(imgs)[3]

    n_batch = sample_shape[0]
    xlen_ = sample_shape[1]
    ylen_ = sample_shape[2]
    
    x = tf.to_float(x)
    y = tf.to_float(y)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    base = _repeat(tf.range(n_batch) * xlen_ * ylen_, ylen_ * xlen_)
    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen
    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])

    # reshape
    output = tf.reshape(output, [n_batch, xlen_, ylen_, n_channel])

    return output

def _interpolate3d(imgs, x, y, z, sample_shape):
    # n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[4]

    n_batch = sample_shape[0]
    xlen_ = sample_shape[1]
    ylen_ = sample_shape[2]
    zlen_ = sample_shape[3]

    x = tf.to_float(x)
    y = tf.to_float(y)
    z = tf.to_float(z)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zlen_f = tf.to_float(zlen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    max_z = tf.cast(zlen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)
    base = _repeat(tf.range(n_batch) * xlen_ * ylen_ * zlen_,
                   xlen_ * ylen_ * zlen_)
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    dz = z - tf.to_float(z0)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    output = tf.reshape(output, [n_batch, xlen_, ylen_, zlen_, n_channel])
    # output = tf.concat([output]*n, axis=0)
    return output

def batch_affine_warp2d(imgs, theta):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    theta = tf.reshape(theta, [-1, 2, 3])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])

    T_g = tf.matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    with tf.Session() as sess:
        print(sess.run(T_g))

    output = batch_warp2d(imgs, T_g)
    return output


def batch_affine_warp3d(imgs, theta):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = batch_warp3d(imgs, T_g)
    return output

def grad(p):
    dx = p[:,:,:,1:] - p[:,:,:,:-1]
    dy = p[:,:,1:,:] - p[:,:,:-1,:]
    dz = p[:,1:,:,:] - p[:,:-1,:,:]
    dx = tf.concat((dx, tf.expand_dims(dx[:,:,:,-1], axis=3)), axis=3)
    dy = tf.concat((dy, tf.expand_dims(dy[:,:,-1,:], axis=2)), axis=2)
    dz = tf.concat((dz, tf.expand_dims(dz[:,-1,:,:], axis=1)), axis=1)
    return tf.concat([dx,dy,dz], axis=-1)    

def curl(s):
    # s: [B,D H,W,3]
    # dudx = s[:,:,:,1:,0] - s[:,:,:,:-1,0]
    dvdx = s[:,:,:,1:,1] - s[:,:,:,:-1,1]
    dwdx = s[:,:,:,1:,2] - s[:,:,:,:-1,2]
    
    dudy = s[:,:,1:,:,0] - s[:,:,:-1,:,0]
    # dvdy = s[:,:,1:,:,1] - s[:,:,:-1,:,1]
    dwdy = s[:,:,1:,:,2] - s[:,:,:-1,:,2]
    
    dudz = s[:,1:,:,:,0] - s[:,:-1,:,:,0]
    dvdz = s[:,1:,:,:,1] - s[:,:-1,:,:,1]
    # dwdz = s[:,1:,:,:,2] - s[:,:-1,:,:,2]

    # dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = tf.concat((dvdx, tf.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = tf.concat((dwdx, tf.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    # dvdy = tf.concat((dvdy, tf.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = tf.concat((dwdy, tf.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = tf.concat((dvdz, tf.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    # dwdz = tf.concat((dwdz, tf.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    return tf.stack([u,v,w], axis=-1)

def advect(d, vel, order=1, is_3d=False):
    n_batch = 1 # assert(tf.shape(d)[0] == 1)
    xlen = tf.shape(d)[1]
    ylen = tf.shape(d)[2]
    
    if is_3d:
        zlen = tf.shape(d)[3]
        grids = batch_mgrid(n_batch, xlen, ylen, zlen) # [b,3,u,v,w]
        vel = tf.transpose(vel, [0,4,1,2,3]) # [b,u,v,w,3] -> [b,3,u,v,w]
        grids -= vel # p' = p - v*dt, dt = 1

        if order == 1: # semi-lagrangian
            d_adv = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
        else: # maccormack
            d_fwd = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
            grids_ = batch_mgrid(n_batch, xlen, ylen, zlen) + vel
            d_bwd = batch_warp3d(d_fwd, grids_, [n_batch, xlen, ylen, zlen])
            d_adv = d_fwd + (d-d_bwd)*0.5
            d_max = tf.nn.max_pool3d(d, ksize=(1,2,2,2,1), strides=(1,1,1,1,1), padding='SAME')
            d_min = -tf.nn.max_pool3d(-d, ksize=(1,2,2,2,1), strides=(1,1,1,1,1), padding='SAME')
            grids = tf.to_int32(grids)
            d_max = batch_warp3d(d_max, grids, [n_batch, xlen, ylen, zlen])
            d_min = batch_warp3d(d_min, grids, [n_batch, xlen, ylen, zlen])
            d_max = tf.greater(d_adv, d_max)
            d_min = tf.greater(d_min, d_adv)
            d_adv = tf.where(tf.logical_or(d_min,d_max), d_fwd, d_adv)
    else:
        grids = batch_mgrid(n_batch, xlen, ylen) # [b,2,u,v]
        vel = tf.transpose(vel, [0,3,1,2]) # [b,u,v,2] -> [b,2,u,v]
        grids -= vel # p' = p - v*dt, dt = 1

        if order == 1:
            d_adv = batch_warp2d(d, grids, [n_batch, xlen, ylen])
        else:
            d_fwd = batch_warp2d(d, grids, [n_batch, xlen, ylen])
            grids_ = batch_mgrid(n_batch, xlen, ylen) + vel
            d_bwd = batch_warp2d(d_fwd, grids_, [n_batch, xlen, ylen])
            # flags = tf.clip_by_value(tf.math.ceil(d_fwd), 0, 1)
            d_adv = d_fwd + (d-d_bwd)*0.5
            d_max = tf.nn.max_pool(d, ksize=(1,2,2,1), strides=(1,1,1,1), padding='SAME')
            d_min = -tf.nn.max_pool(-d, ksize=(1,2,2,1), strides=(1,1,1,1), padding='SAME')
            grids = tf.to_int32(grids)
            # d_max = batch_warp2d(d_max, grids, [n_batch, xlen, ylen])
            # d_min = batch_warp2d(d_min, grids, [n_batch, xlen, ylen])
            d_max, d_min = d_max[grids], d_max[grids]
            # # hard clamp
            # d_adv = tf.clip_by_value(d_adv, d_min, d_max)
            # soft clamp
            d_max = tf.greater(d_adv, d_max) # find values larger than max (true if x > y)
            d_min = tf.greater(d_min, d_adv) # find values smaller than min (true if x > y)
            d_adv = tf.where(tf.logical_or(d_min,d_max), d_fwd, d_adv) # *flags
        
    return d_adv

def rotate(d):
    b = tf.shape(d)[0]
    xlen = tf.shape(d)[1]
    ylen = tf.shape(d)[2]
    zlen = tf.shape(d)[3]
    
    rot_mat = tf.placeholder(shape=[None,3,3], dtype=tf.float32)
    n_rot = tf.shape(rot_mat)[0]
    n_batch = b*n_rot

    d = tf.tile(d, [n_rot,1,1,1,1])
    r = tf.tile(rot_mat, [b,1,1])
    grids = batch_mgrid(n_batch, xlen, ylen, zlen) # [b,3,u,v,w]
    grids = tf.reshape(grids, [n_batch, 3, -1])
    grids = tf.matmul(r, grids)
    grids = tf.reshape(grids, [n_batch, 3, xlen, ylen, zlen])
    d_rot = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
    return d_rot, rot_mat

def subsample(d, scale):
    n_batch = tf.shape(d)[0]
    xlen = tf.to_int32(
        tf.multiply(tf.to_float(tf.shape(d)[1]),scale))
    ylen = tf.to_int32(
        tf.multiply(tf.to_float(tf.shape(d)[2]),scale))
    grids = batch_mgrid(n_batch, xlen, ylen) # [b,2,u,v]
    d_sample = batch_warp2d(d, grids, [n_batch, xlen, ylen])
    return d_sample

def rot_z_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]])
    return rot_mat

def rot_y_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c,0,-s],
        [0,1,0],
        [s,0,c]])
    return rot_mat

def rot_x_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [1,0,0],
        [0,c,-s],
        [0,s,c]])
    return rot_mat

def scale(s):
    s_mat = np.array([
        [s,0,0],
        [0,s,0],
        [0,0,s]])
    return s_mat

    
def rot_mat_turb(theta_unit, poisson_sample=False, rng=None):
    views = [{'theta':0}, {'theta':90}, {'theta':180}]
    if poisson_sample:
        views += rot_mat_poisson(0,0,0, 0, 180, theta_unit, rng)

    mat = []
    for view in views:
        theta = view['theta']
        mat.append(rot_y_3d(theta))
    return mat, views

def rot_mat(phi0, phi1, phi_unit, theta0, theta1, theta_unit, 
            sample_type='uniform', rng=None, nv=None):

    if 'uniform' in sample_type:
        views = rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit)
    elif 'poisson' in sample_type:
        views = rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng)
        views += rot_mat_uniform(phi0, phi1, 0, theta0, theta1, 0) # [midpoint]
        if nv is not None:
            if len(views) > nv:
                views = views[len(views)-nv:]
            elif len(views) < nv:
                views_ = rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng)
                views += views_[:nv-len(views)]
    else: # both
        views = rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit)
        views += rot_mat_poisson(phi0, phi1, phi_unit*2, theta0, theta1, theta_unit*2, rng)
        if nv is not None:
            if len(views) > nv:
                views = views[len(views)-nv:]
            elif len(views) < nv:
                views_ = rot_mat_poisson(phi0, phi1, phi_unit*2, theta0, theta1, theta_unit*2, rng)
                views += views_[:nv-len(views)]

    mat = []
    for view in views:
        phi, theta = view['phi'], view['theta']
        rz = rot_z_3d(phi)
        ry = rot_y_3d(theta)
        rot_mat = np.matmul(ry,rz)
        # s = scale(3)
        # rot_mat = np.matmul(s, rot_mat)
        mat.append(rot_mat)
    return mat, views

def rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng):
    if phi_unit == 0:
        h = 1
        phi0 = -0.5
    else:
        h = phi1 - phi0

    if theta_unit == 0:
        w = 1
        theta0 = -0.5
    else:
        w = theta1 - theta0

    r = max(phi_unit, theta_unit)/2

    p = PoissonDisc(rng, height=h, width=w, r=r)
    s = p.sample()

    views = []
    for s_ in s:
        phi_ = s_[1]+phi0
        theta_ = s_[0]+theta0
        views.append({'phi':phi_, 'theta':theta_})

    return views

def rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit):
    if phi_unit == 0:
        phi = [(phi1-phi0)/2]
    else:
        n_phi = np.abs(phi1-phi0) / float(phi_unit) + 1
        phi = np.linspace(phi0, phi1, n_phi, endpoint=True)

    if theta_unit == 0:
        theta = [(theta1-theta0)/2]
    else:
        n_theta = np.abs(theta1-theta0) / float(theta_unit) + 1
        theta = np.linspace(theta0, theta1, n_theta, endpoint=True)    

    views = []
    for phi_ in phi:
        for theta_ in theta:
            views.append({'phi':phi_, 'theta':theta_})

    return views    

if __name__ == '__main__':
    """
    for test

    the result will be

    the original image
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    identity warped
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    zoom in warped
    [[  6.    6.5   7.    7.5   8. ]
     [  8.5   9.    9.5  10.   10.5]
     [ 11.   11.5  12.   12.5  13. ]
     [ 13.5  14.   14.5  15.   15.5]
     [ 16.   16.5  17.   17.5  18. ]]
    """
    img = tf.to_float(np.arange(25).reshape(1, 5, 5, 1))
    identity_matrix = tf.to_float([1, 0, 0, 0, 1, 0])
    zoom_in_matrix = identity_matrix * 0.5
    identity_warped = batch_affine_warp2d(img, identity_matrix)
    zoom_in_warped = batch_affine_warp2d(img, zoom_in_matrix)
    with tf.Session() as sess:
        print(sess.run(img[0, :, :, 0]))

        # # mgrid test
        # print(sess.run(batch_mgrid(2, 5, 4)))

        print(sess.run(identity_warped[0, :, :, 0]))
        print(sess.run(zoom_in_warped[0, :, :, 0]))