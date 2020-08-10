#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################

# https://github.com/Ryo-Ito/spatial_transformer_network

import tensorflow as tf
import numpy as np

# https://github.com/scipython/scipython_maths/blob/master/poisson_disc_sampled_noise/poisson.py
# For mathematical details of this algorithm, please see the blog
# article at https://scipython.com/blog/poisson-disc-sampling-in-python/
# Christian Hill, March 2017.
class PoissonDisc(object):
    """A class for generating two-dimensional Possion (blue) noise)."""

    def __init__(self, rng, width=50, height=50, r=1, k=30):
        self.rng = rng
        self.width, self.height = width, height
        self.r = r
        self.k = k

        # Cell side length
        self.a = r/np.sqrt(2)
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        self.reset()

    def reset(self):
        """Reset the cells dictionary."""

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(self.nx)
                                for iy in range(self.ny)]
        # Initilalize the dictionary of cells: each key is a cell's coordinates
        # the corresponding value is the index of that cell's point's
        # coordinates in the samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] // self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in the
        cells with neighbouring coordinates illustrated below: ie those cells
        that could contain points closer than r.
                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo
        """
        
        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store the index of the contained point
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?
        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.
        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < self.r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True
    
    def get_point(self, refpt):
        """Try to find a candidate point near refpt to emit in the sample.
        We draw up to k points from the annulus of inner radius r, outer radius
        2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return
        False. Otherwise, return the pt.
        """

        i = 0
        while i < self.k:
            rho, theta = (self.rng.uniform(self.r, 2*self.r),
                          self.rng.uniform(0, 2*np.pi))
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 < pt[0] < self.width and 0 < pt[1] < self.height):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self):
        """Poisson disc random sampling in 2D.
        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.
        """

        # Pick a random point to start with.
        pt = (self.rng.uniform(0, self.width),
              self.rng.uniform(0, self.height))
        self.samples = [pt]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self.get_cell_coords(pt)] = 0
        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = self.rng.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                self.samples.append(pt)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self.get_cell_coords(pt)] = nsamples
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)

        return self.samples

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
    low = tf.cast(low, tf.float32)
    high = tf.cast(high, tf.float32)
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
    #     tf.reshape(tf.cast(base_indices, tf.float32), [-1, 1]),
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
    
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
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
    imgs_flat = tf.cast(imgs_flat, tf.float32)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.cast(x0, tf.float32)
    dy = y - tf.cast(y0, tf.float32)
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

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
    zlen_f = tf.cast(zlen, tf.float32)
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
    imgs_flat = tf.cast(imgs_flat, tf.float32)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.cast(x0, tf.float32)
    dy = y - tf.cast(y0, tf.float32)
    dz = z - tf.cast(z0, tf.float32)
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

def curl(s, is_2d=True):
    if is_2d:
        # s: [B,H,W,1]
        u = s[:,1:,:,0] - s[:,:-1,:,0] # ds/dy
        v = s[:,:,:-1,0] - s[:,:,1:,0] # -ds/dx,
        u = tf.concat([u, tf.expand_dims(u[:,-1,:], axis=1)], axis=1)
        v = tf.concat([v, tf.expand_dims(v[:,:,-1], axis=2)], axis=2)
        return tf.stack([u,v], axis=-1)
    else:
        # s: [B,D,H,W,3]
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
        tf.multiply(tf.cast(tf.shape(d)[1], tf.float32),scale))
    ylen = tf.to_int32(
        tf.multiply(tf.cast(tf.shape(d)[2], tf.float32),scale))
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
    

def g2p(g, p, is_2d=True, is_linear=False):

    if is_linear:
        return g2p_linear(g, p, is_2d)
    else:
        return g2p_cubic(g, p, is_2d)

def g2p_cubic(g, p, is_2d=True):
    n_batch = 1 # tf.shape(g)[0]
    xlen = tf.shape(g)[1]
    ylen = tf.shape(g)[2]
    if is_2d:
        n_channel = tf.shape(g)[3]
    else:
        zlen = tf.shape(g)[3]
        n_channel = tf.shape(g)[4]
    pn = tf.shape(p)[1]
    
    x = tf.cast(p[0,...,0], tf.float32) # [0-1]
    y = tf.cast(p[0,...,1], tf.float32)
    if not is_2d:
        z = tf.cast(p[0,...,2], tf.float32)
    
    # scale to g
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
    x *= xlen_f    
    y *= ylen_f
    if not is_2d:
        zlen_f = tf.cast(zlen, tf.float32)
        z *= zlen_f
    
    # do sampling
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    if not is_2d:
        max_z = tf.cast(zlen - 1, 'int32')
    
    # shifted index to interpolate cell centers
    x1 = tf.cast(tf.floor(x - 0.5), 'int32')
    x0 = x1 - 1
    x2 = x1 + 1
    x3 = x1 + 2
    y1 = tf.cast(tf.floor(y - 0.5), 'int32')
    y0 = y1 - 1
    y2 = y1 + 1
    y3 = y1 + 2
    if not is_2d:
        z1 = tf.cast(tf.floor(z - 0.5), 'int32')
        z0 = z1 - 1
        z2 = z1 + 1
        z3 = z1 + 2

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    x2 = tf.clip_by_value(x2, zero, max_x)
    x3 = tf.clip_by_value(x3, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    y2 = tf.clip_by_value(y2, zero, max_y)
    y3 = tf.clip_by_value(y3, zero, max_y)
    if not is_2d:
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)
        z2 = tf.clip_by_value(z2, zero, max_z)
        z3 = tf.clip_by_value(z3, zero, max_z)
    
    # compute flat indices
    if is_2d:
        base = _repeat(tf.range(n_batch)*xlen*ylen, pn)
        base_x0 = base + x0 * ylen
        base_x1 = base + x1 * ylen
        base_x2 = base + x2 * ylen
        base_x3 = base + x3 * ylen

        index00 = base_x0 + y0
        index01 = base_x0 + y1
        index02 = base_x0 + y2
        index03 = base_x0 + y3
        index10 = base_x1 + y0
        index11 = base_x1 + y1
        index12 = base_x1 + y2
        index13 = base_x1 + y3
        index20 = base_x2 + y0
        index21 = base_x2 + y1
        index22 = base_x2 + y2
        index23 = base_x2 + y3
        index30 = base_x3 + y0
        index31 = base_x3 + y1
        index32 = base_x3 + y2
        index33 = base_x3 + y3
    else:
        base = _repeat(tf.range(n_batch)*xlen*ylen*zlen, pn)
        base_x0 = base + x0 * ylen * zlen
        base_x1 = base + x1 * ylen * zlen
        base_x2 = base + x2 * ylen * zlen
        base_x3 = base + x3 * ylen * zlen

        base00 = base_x0 + y0 * zlen
        base01 = base_x0 + y1 * zlen
        base02 = base_x0 + y2 * zlen
        base03 = base_x0 + y3 * zlen
        base10 = base_x1 + y0 * zlen
        base11 = base_x1 + y1 * zlen
        base12 = base_x1 + y2 * zlen
        base13 = base_x1 + y3 * zlen
        base20 = base_x2 + y0 * zlen
        base21 = base_x2 + y1 * zlen
        base22 = base_x2 + y2 * zlen
        base23 = base_x2 + y3 * zlen
        base30 = base_x3 + y0 * zlen
        base31 = base_x3 + y1 * zlen
        base32 = base_x3 + y2 * zlen
        base33 = base_x3 + y3 * zlen

        index000 = base00 + z0
        index001 = base00 + z1
        index002 = base00 + z2
        index003 = base00 + z3

        index010 = base01 + z0
        index011 = base01 + z1
        index012 = base01 + z2
        index013 = base01 + z3

        index020 = base02 + z0
        index021 = base02 + z1
        index022 = base02 + z2
        index023 = base02 + z3

        index030 = base03 + z0
        index031 = base03 + z1
        index032 = base03 + z2
        index033 = base03 + z3

        index100 = base10 + z0
        index101 = base10 + z1
        index102 = base10 + z2
        index103 = base10 + z3

        index110 = base11 + z0
        index111 = base11 + z1
        index112 = base11 + z2
        index113 = base11 + z3

        index120 = base12 + z0
        index121 = base12 + z1
        index122 = base12 + z2
        index123 = base12 + z3

        index130 = base13 + z0
        index131 = base13 + z1
        index132 = base13 + z2
        index133 = base13 + z3

        index200 = base20 + z0
        index201 = base20 + z1
        index202 = base20 + z2
        index203 = base20 + z3

        index210 = base21 + z0
        index211 = base21 + z1
        index212 = base21 + z2
        index213 = base21 + z3

        index220 = base22 + z0
        index221 = base22 + z1
        index222 = base22 + z2
        index223 = base22 + z3

        index230 = base23 + z0
        index231 = base23 + z1
        index232 = base23 + z2
        index233 = base23 + z3

        index300 = base30 + z0
        index301 = base30 + z1
        index302 = base30 + z2
        index303 = base30 + z3

        index310 = base31 + z0
        index311 = base31 + z1
        index312 = base31 + z2
        index313 = base31 + z3

        index320 = base32 + z0
        index321 = base32 + z1
        index322 = base32 + z2
        index323 = base32 + z3

        index330 = base33 + z0
        index331 = base33 + z1
        index332 = base33 + z2
        index333 = base33 + z3

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    g_flat = tf.reshape(g, [-1, n_channel])
    g_flat = tf.cast(g_flat, tf.float32)

    def _hermite(A, B, C, D, t):
        # https://github.com/iwyoo/bicubic_interp-tensorflow/blob/master/bicubic_interp.py
        a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
        b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
        c = A * (-0.5) + C * 0.5
        d = B
        return a*t*t*t + b*t*t + c*t + d
        
    if is_2d:    
        I00 = tf.gather(g_flat, index00)
        I01 = tf.gather(g_flat, index01)
        I02 = tf.gather(g_flat, index02)
        I03 = tf.gather(g_flat, index03)
        I10 = tf.gather(g_flat, index10)
        I11 = tf.gather(g_flat, index11)
        I12 = tf.gather(g_flat, index12)
        I13 = tf.gather(g_flat, index13)
        I20 = tf.gather(g_flat, index20)
        I21 = tf.gather(g_flat, index21)
        I22 = tf.gather(g_flat, index22)
        I23 = tf.gather(g_flat, index23)
        I30 = tf.gather(g_flat, index30)
        I31 = tf.gather(g_flat, index31)
        I32 = tf.gather(g_flat, index32)
        I33 = tf.gather(g_flat, index33)

        # and finally calculate interpolated values
        dx = x - (tf.cast(x1, tf.float32) + 0.5)
        dx = tf.expand_dims(dx, axis=-1)
        I0 = _hermite(I00, I10, I20, I30, dx)
        I1 = _hermite(I01, I11, I21, I31, dx)
        I2 = _hermite(I02, I12, I22, I32, dx)
        I3 = _hermite(I03, I13, I23, I33, dx)

        dy = y - (tf.cast(y1, tf.float32) + 0.5)
        dy = tf.expand_dims(dy, axis=-1)
        output = _hermite(I0, I1, I2, I3, dy)
    else:
        I000 = tf.gather(g_flat, index000)
        I001 = tf.gather(g_flat, index001)
        I002 = tf.gather(g_flat, index002)
        I003 = tf.gather(g_flat, index003)
        I010 = tf.gather(g_flat, index010)
        I011 = tf.gather(g_flat, index011)
        I012 = tf.gather(g_flat, index012)
        I013 = tf.gather(g_flat, index013)
        I020 = tf.gather(g_flat, index020)
        I021 = tf.gather(g_flat, index021)
        I022 = tf.gather(g_flat, index022)
        I023 = tf.gather(g_flat, index023)
        I030 = tf.gather(g_flat, index030)
        I031 = tf.gather(g_flat, index031)
        I032 = tf.gather(g_flat, index032)
        I033 = tf.gather(g_flat, index033)
        I100 = tf.gather(g_flat, index100)
        I101 = tf.gather(g_flat, index101)
        I102 = tf.gather(g_flat, index102)
        I103 = tf.gather(g_flat, index103)
        I110 = tf.gather(g_flat, index110)
        I111 = tf.gather(g_flat, index111)
        I112 = tf.gather(g_flat, index112)
        I113 = tf.gather(g_flat, index113)
        I120 = tf.gather(g_flat, index120)
        I121 = tf.gather(g_flat, index121)
        I122 = tf.gather(g_flat, index122)
        I123 = tf.gather(g_flat, index123)
        I130 = tf.gather(g_flat, index130)
        I131 = tf.gather(g_flat, index131)
        I132 = tf.gather(g_flat, index132)
        I133 = tf.gather(g_flat, index133)
        I200 = tf.gather(g_flat, index200)
        I201 = tf.gather(g_flat, index201)
        I202 = tf.gather(g_flat, index202)
        I203 = tf.gather(g_flat, index203)
        I210 = tf.gather(g_flat, index210)
        I211 = tf.gather(g_flat, index211)
        I212 = tf.gather(g_flat, index212)
        I213 = tf.gather(g_flat, index213)
        I220 = tf.gather(g_flat, index220)
        I221 = tf.gather(g_flat, index221)
        I222 = tf.gather(g_flat, index222)
        I223 = tf.gather(g_flat, index223)
        I230 = tf.gather(g_flat, index230)
        I231 = tf.gather(g_flat, index231)
        I232 = tf.gather(g_flat, index232)
        I233 = tf.gather(g_flat, index233)
        I300 = tf.gather(g_flat, index300)
        I301 = tf.gather(g_flat, index301)
        I302 = tf.gather(g_flat, index302)
        I303 = tf.gather(g_flat, index303)
        I310 = tf.gather(g_flat, index310)
        I311 = tf.gather(g_flat, index311)
        I312 = tf.gather(g_flat, index312)
        I313 = tf.gather(g_flat, index313)
        I320 = tf.gather(g_flat, index320)
        I321 = tf.gather(g_flat, index321)
        I322 = tf.gather(g_flat, index322)
        I323 = tf.gather(g_flat, index323)
        I330 = tf.gather(g_flat, index330)
        I331 = tf.gather(g_flat, index331)
        I332 = tf.gather(g_flat, index332)
        I333 = tf.gather(g_flat, index333)

        # and finally calculate interpolated values
        dx = x - (tf.cast(x1, tf.float32) + 0.5)
        dx = tf.expand_dims(dx, axis=-1)
        I00 = _hermite(I000, I100, I200, I300, dx)
        I01 = _hermite(I001, I101, I201, I301, dx)
        I02 = _hermite(I002, I102, I202, I302, dx)
        I03 = _hermite(I003, I103, I203, I303, dx)
        I10 = _hermite(I010, I110, I210, I310, dx)
        I11 = _hermite(I011, I111, I211, I311, dx)
        I12 = _hermite(I012, I112, I212, I312, dx)
        I13 = _hermite(I013, I113, I213, I313, dx)
        I20 = _hermite(I020, I120, I220, I320, dx)
        I21 = _hermite(I021, I121, I221, I321, dx)
        I22 = _hermite(I022, I122, I222, I322, dx)
        I23 = _hermite(I023, I123, I223, I323, dx)
        I30 = _hermite(I030, I130, I230, I330, dx)
        I31 = _hermite(I031, I131, I231, I331, dx)
        I32 = _hermite(I032, I132, I232, I332, dx)
        I33 = _hermite(I033, I133, I233, I333, dx)

        dy = y - (tf.cast(y1, tf.float32) + 0.5)
        dy = tf.expand_dims(dy, axis=-1)
        I0 = _hermite(I00, I10, I20, I30, dy)
        I1 = _hermite(I01, I11, I21, I31, dy)
        I2 = _hermite(I02, I12, I22, I32, dy)
        I3 = _hermite(I03, I13, I23, I33, dy)

        dz = z - (tf.cast(z1, tf.float32) + 0.5)
        dz = tf.expand_dims(dz, axis=-1)
        output = _hermite(I0, I1, I2, I3, dz)

    # reshape
    output = tf.reshape(output, [n_batch, pn, n_channel])
    return output

def g2p_linear(g, p, is_2d=True):
    n_batch = 1 # tf.shape(g)[0]
    xlen = tf.shape(g)[1]
    ylen = tf.shape(g)[2]
    if is_2d:
        n_channel = tf.shape(g)[3]
    else:
        zlen = tf.shape(g)[3]
        n_channel = tf.shape(g)[4]
    pn = tf.shape(p)[1]
    
    x = tf.cast(p[0,...,0], tf.float32) # [0-1]
    y = tf.cast(p[0,...,1], tf.float32)
    if not is_2d:
        z = tf.cast(p[0,...,2], tf.float32)
    
    # scale to g
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
    x *= xlen_f    
    y *= ylen_f
    if not is_2d:
        zlen_f = tf.cast(zlen, tf.float32)
        z *= zlen_f
    
    # do sampling
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    if not is_2d:
        max_z = tf.cast(zlen - 1, 'int32')
    
    # shifted index to interpolate cell centers
    x0 = tf.cast(tf.floor(x - 0.5), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y - 0.5), 'int32')
    y1 = y0 + 1
    if not is_2d:
        z0 = tf.cast(tf.floor(z - 0.5), 'int32')
        z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    if not is_2d:
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)
    
    # compute flat indices
    if is_2d:
        base = _repeat(tf.range(n_batch)*xlen*ylen, pn)
        base_x0 = base + x0 * ylen
        base_x1 = base + x1 * ylen
        index00 = base_x0 + y0
        index01 = base_x0 + y1
        index10 = base_x1 + y0
        index11 = base_x1 + y1
    else:
        base = _repeat(tf.range(n_batch)*xlen*ylen*zlen, pn)
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
    g_flat = tf.reshape(g, [-1, n_channel])
    g_flat = tf.cast(g_flat, tf.float32)

    if is_2d:    
        I00 = tf.gather(g_flat, index00)
        I01 = tf.gather(g_flat, index01)
        I10 = tf.gather(g_flat, index10)
        I11 = tf.gather(g_flat, index11)

        # and finally calculate interpolated values
        dx = x - (tf.cast(x0, tf.float32) + 0.5)
        dy = y - (tf.cast(y0, tf.float32) + 0.5)
        w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
        w01 = tf.expand_dims((1. - dx) * dy, 1)
        w10 = tf.expand_dims(dx * (1. - dy), 1)
        w11 = tf.expand_dims(dx * dy, 1)
        output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    else:
        I000 = tf.gather(g_flat, index000)
        I001 = tf.gather(g_flat, index001)
        I010 = tf.gather(g_flat, index010)
        I011 = tf.gather(g_flat, index011)
        I100 = tf.gather(g_flat, index100)
        I101 = tf.gather(g_flat, index101)
        I110 = tf.gather(g_flat, index110)
        I111 = tf.gather(g_flat, index111)

        # and finally calculate interpolated values
        dx = x - (tf.cast(x0, tf.float32) + 0.5)
        dy = y - (tf.cast(y0, tf.float32) + 0.5)
        dz = z - (tf.cast(z0, tf.float32) + 0.5)
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
    output = tf.reshape(output, [n_batch, pn, n_channel])
    return output

def W(k='cubic'):
    def cubicspline(q, h, is_3d=False):
        if is_3d:
            sigma = 8/np.pi/(h**3)
        else:
            sigma = 40/7/np.pi/(h**2)
        return tf.compat.v1.where(q > 1,
            tf.zeros_like(q),
            sigma * tf.where(q <= 0.5,
                6 * (q**3 - q**2) + 1,
                2 * (1-q)**3
                )
        )

    def linear(q, h=0, is_3d=False):
        return tf.maximum(1 - q, 0)

    def smooth(q, h=0, is_3d=False):
        return tf.maximum(1 - q**2, 0)
    
    def sharp(q, h=0, is_3d=False):
        return tf.maximum((1/q)**2 - 1, 0)

    def poly6(q, h, is_3d=False):
        if is_3d:
            sigma = 315/64/np.pi/h**9
        else:
            sigma = 4/np.pi/h**8
        return tf.maximum(sigma*(h**2 - q**2)**3, 0)

    if k == 'cubic': return cubicspline
    elif k == 'linear': return linear
    elif k == 'smooth': return smooth
    elif k == 'sharp': return sharp
    elif k == 'poly6': return poly6

def GW(k='cubic'):
    def cubicspline(r, h, is_3d=False):
        if is_3d:
            sigma = 48/np.pi/(h**3)
        else:
            sigma = 240/7/np.pi/(h**2)
        rl = tf.sqrt(tf.reduce_sum(r**2, axis=-1))
        rl = tf.tile(tf.expand_dims(rl, axis=-1), [1,1,r.shape[-1]])
        q = rl / h
        return tf.compat.v1.where(tf.logical_and(q <= 1, rl > 1e-6),
            sigma / (rl * h) * r * tf.where(q <= 0.5,
                q * (3*q - 2),
                -(1-q)**2
                ),
            tf.zeros_like(r),
        )

    if k == 'cubic': return cubicspline

# MIT License

# Copyright (c) 2018 Eldar Insafutdinov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#  https://github.com/eldar/differentiable-point-clouds/dpc/util/point_cloud.py-pointcloud2voxel3d_fast
def p2g(p, domain, res, radius, rest_density, nsize, pc=None, pd=None, is_2d=True, kernel='cubic', eps=1e-6, clip=True, support=4):
    # p.shape: [B,N,2 or 3]    
    batch_size = p.shape[0]
    num_points = tf.shape(p)[1]

    # scale [0,1] -> [domain size]
    domain_ = tf.cast(domain, tf.float32)
    p = p * domain_
    
    # clip for outliers (after advection)
    if clip:
        p = tf.clip_by_value(p, 0, domain_ - eps)
    else:
        valid = tf.logical_and(p >= 0, p < domain_)
        valid = tf.reduce_all(valid, axis=-1)
        valid = tf.reshape(valid, [-1])

    # compute grid id
    cell_size = domain_ / tf.cast(res, tf.float32)
    # assert cell_size[0] == cell_size[1]
    cell_size = cell_size[0]
    indices_floor = tf.floor(p / cell_size)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)
    
    gc = (indices_floor+0.5)*cell_size # grid cell center
    r = p - gc # fractional part from grid cell center
    rr = []
    for n in range(-nsize,nsize+1):
        rr.append(r-n*cell_size) # [+cx,0,-cx]
    # rr = [r]

    # for sph
    W_ = W(kernel)
    support_radius = radius*support # 2*particle spacing
    if is_2d:
        volume = 0.8 * (2*radius)**2
    else:
        volume = 0.8 * (2*radius)**3
    mass = volume * rest_density

    if is_2d:
        # [B,N,3], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 3])

        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter2d(pos):
            dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1]
            q = tf.sqrt(dx**2 + dy**2) / support_radius

            updates_raw = W_(q, support_radius, is_3d=False)
            updates = mass*tf.reshape(updates_raw, [-1])
            if not clip:
                updates = tf.boolean_mask(updates, valid)

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize]])
            # indices_shift = tf.constant([[0] + [pos[0], pos[1]]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            if pc is None:
                img = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res), axis=-1)) #[batch_size]+res
                img = tf.expand_dims(img, axis=-1)
            else:
                updates_ = mass*tf.expand_dims(updates_raw, axis=-1) * pc
                if pd is None:
                    updates_ /= rest_density
                else:
                    updates_ /= pd
                updates = tf.reshape(updates_, [-1, tf.shape(pc)[-1]])
                if not clip:
                    updates = tf.boolean_mask(updates, valid)
                img = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[tf.shape(pc)[-1]]), axis=-1))
            
            return img

        img = []
        for j in range(2*nsize+1):
            for i in range(2*nsize+1):
                vx = interpolate_scatter2d([j, i])
                img.append(vx)

        # vx, vx_rgb = interpolate_scatter2d([0, 0])
        # img.append(vx)
        # img_rgb.append(vx_rgb)

        img = tf.add_n(img)[:,::-1] # flip in y
        return img
    else:
        # [B,N,4], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 4])

        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter3d(pos):
            dz,dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1], rr[pos[2]][...,2]
            q = tf.sqrt(dx**2 + dy**2 + dz**2) / support_radius

            updates_raw = W_(q, support_radius, is_3d=True)
            updates = mass*tf.reshape(updates_raw, [-1])
            if not clip:
                updates = tf.boolean_mask(updates, valid)

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize, pos[2]-nsize]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            if pc is None:
                vox = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res), axis=-1))
                vox = tf.expand_dims(vox, axis=-1)
            else:
                updates_ = mass*tf.expand_dims(updates_raw, axis=-1) * pc
                if pd is None:
                    updates_ /= rest_density
                else:
                    updates_ /= pd
                updates = tf.reshape(updates_, [-1, tf.shape(pc)[-1]])
                if not clip:
                    updates = tf.boolean_mask(updates, valid)
                vox = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[tf.shape(pc)[-1]]), axis=-1))
            
            return vox

        vox = []
        for k in range(2*nsize+1):
            for j in range(2*nsize+1):
                for i in range(2*nsize+1):
                    vx = interpolate_scatter3d([k,j,i])
                    vox.append(vx)

        vox = tf.add_n(vox)[:,:,::-1] # flip in y
        return vox

def p2g_grad(p, domain, res, radius, rest_density, nsize, pc=None, pd=None, is_2d=True, kernel='cubic', eps=1e-6, clip=True):
    # p.shape: [B,N,2 or 3]    
    batch_size = p.shape[0]
    num_points = tf.shape(p)[1]

    # scale [0,1] -> [domain size]
    domain_ = tf.cast(domain, tf.float32)
    p = p * domain_
    
    # clip for outliers (after advection)
    if clip:
        p = tf.clip_by_value(p, 0, domain_ - eps)
    else:
        valid = tf.logical_and(p >= 0, p < domain_)
        valid = tf.reduce_all(valid, axis=-1)
        valid = tf.reshape(valid, [-1])

    # compute grid id
    cell_size = domain_ / tf.cast(res, tf.float32)
    # assert cell_size[0] == cell_size[1]
    cell_size = cell_size[0]
    indices_floor = tf.floor(p / cell_size)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)
    
    gc = (indices_floor+0.5)*cell_size # grid cell center
    r = gc - p # fractional part from grid cell center
    rr = []
    for n in range(-nsize,nsize+1):
        rr.append(r+n*cell_size) # [+cx,0,-cx]
    # rr = [r]

    # for sph
    W_ = GW(kernel)
    support_radius = radius*4 # 2*particle spacing
    if is_2d:
        volume = 0.8 * (2*radius)**2
    else:
        volume = 0.8 * (2*radius)**3
    mass = volume * rest_density

    if is_2d:
        # [B,N,3], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 3])

        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter2d(pos):
            dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1]
            updates_raw = W_(tf.stack([dy,dx], axis=-1), support_radius, is_3d=False)

            updates = mass*updates_raw
            if pd is None:
                updates /= rest_density
            else:
                updates /= pd
            updates = tf.reshape(updates, [-1, 2])
            if not clip:
                updates = tf.boolean_mask(updates, valid)

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            n = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[2]), axis=-1))
            return n

        n = []
        for j in range(2*nsize+1):
            for i in range(2*nsize+1):
                n_ = interpolate_scatter2d([j, i])
                n.append(n_)

        n = -tf.add_n(n)[:,::-1] # flip in y
        return n
    else:
        # [B,N,4], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 4])

        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter3d(pos):
            dz,dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1], rr[pos[2]][...,2]
            updates_raw = W_(tf.stack([dz,dy,dx], axis=-1), support_radius, is_3d=True)

            updates = mass*updates_raw
            if pd is None:
                updates /= rest_density
            else:
                updates /= pd
            updates = tf.reshape(updates, [-1, 3])
            if not clip:
                updates = tf.boolean_mask(updates, valid)

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize, pos[2]-nsize]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            n = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[3]), axis=-1))
            return n

        n = []
        for k in range(2*nsize+1):
            for j in range(2*nsize+1):
                for i in range(2*nsize+1):
                    n_ = interpolate_scatter3d([k,j,i])
                    n.append(n_)

        n = -tf.add_n(n)[:,:,::-1] # flip in y
        return n

def p2g_wavg(p, x, domain, res, radius, nsize, is_2d=True, kernel='linear', eps=1e-6, clip=True, support=4):
    # p.shape: [B,N,2 or 3]    
    batch_size = p.shape[0]
    num_points = tf.shape(p)[1]

    # scale [0,1] -> [domain size]
    domain_ = tf.cast(domain, tf.float32)
    p = p * domain_
    
    # clip for outliers (after advection)
    if clip:
        p = tf.clip_by_value(p, 0, domain_ - eps)
    else:
        valid = tf.logical_and(p >= 0, p < domain_)
        valid = tf.reduce_all(valid, axis=-1)
        valid = tf.reshape(valid, [-1])

    # compute grid id
    cell_size = domain_ / tf.cast(res, tf.float32)
    # assert cell_size[0] == cell_size[1]
    cell_size = cell_size[0]
    indices_floor = tf.floor(p / cell_size)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)
    
    gc = (indices_floor+0.5)*cell_size # grid cell center
    r = p - gc # fractional part from grid cell center
    rr = []
    for n in range(-nsize,nsize+1):
        rr.append(r-n*cell_size) # [+cx,0,-cx]
    # rr = [r]

    W_ = W(kernel)
    support_radius = radius*support # 2*particle spacing

    if is_2d:
        # [B,N,3], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 3])
        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter2d(pos):
            dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1]
            q = tf.sqrt(dx**2 + dy**2) / support_radius

            updates_raw = W_(q, support_radius, is_3d=False)
            updates = tf.reshape(updates_raw, [-1])
            if not clip:
                updates = tf.boolean_mask(updates, valid)
            
            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize]])
            # indices_shift = tf.constant([[0] + [pos[0], pos[1]]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            wmap = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res), axis=-1)) #[batch_size]+res
            wmap = tf.expand_dims(wmap, axis=-1)
            wmap = tf.tile(wmap, [1, 1, 1, tf.shape(x)[-1]])

            updates_x = tf.expand_dims(updates_raw, axis=-1) * x
            updates = tf.reshape(updates_x, [-1, tf.shape(x)[-1]])
            if not clip:
                updates = tf.boolean_mask(updates, valid)
            d_img = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[tf.shape(x)[-1]]), axis=-1))
            return wmap, d_img

        wmap, d_img = [], []
        for j in range(2*nsize+1):
            for i in range(2*nsize+1):
                w, d = interpolate_scatter2d([j, i])
                wmap.append(w)
                d_img.append(d)

        wmap = tf.add_n(wmap)[:,::-1] # flip
        d_img = tf.add_n(d_img)[:,::-1] # flip
        d_img = tf.compat.v1.where(wmap > eps, d_img/wmap, d_img)
        return d_img
    else:
        # [B,N,4], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 4])
        if not clip:
            indices = tf.boolean_mask(indices, valid)

        def interpolate_scatter3d(pos):
            dz,dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1], rr[pos[2]][...,2]
            q = tf.sqrt(dx**2 + dy**2 + dz**2) / support_radius

            updates_raw = W_(q, support_radius, is_3d=True)
            updates = tf.reshape(updates_raw, [-1])
            if not clip:
                updates = tf.boolean_mask(updates, valid)

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize, pos[2]-nsize]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            wmap = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res), axis=-1)) #[batch_size]+res
            wmap = tf.expand_dims(wmap, axis=-1)
            wmap = tf.tile(wmap, [1, 1, 1, 1, tf.shape(x)[-1]])

            updates_x = tf.expand_dims(updates_raw, axis=-1) * x
            updates = tf.reshape(updates_x, [-1, tf.shape(x)[-1]])
            if not clip:
                updates = tf.boolean_mask(updates, valid)
            d_vox = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[tf.shape(x)[-1]]), axis=-1))
            return wmap, d_vox

        wmap, d_vox = [], []
        for k in range(2*nsize+1):
            for j in range(2*nsize+1):
                for i in range(2*nsize+1):
                    w, vx = interpolate_scatter3d([k,j,i])
                    wmap.append(w)
                    d_vox.append(vx)

        wmap = tf.add_n(wmap)[:,:,::-1] # flip
        d_vox = tf.add_n(d_vox)[:,:,::-1] # flip
        d_vox = tf.compat.v1.where(wmap > eps, d_vox/wmap, d_vox)
        return d_vox

def p2g_repulsive(p, domain, res, radius, nsize, is_2d=True, kernel='smooth', eps=1e-6, alpha=50):
    # p.shape: [B,N,2 or 3]    
    batch_size = p.shape[0]
    num_points = tf.shape(p)[1]

    # scale [0,1] -> [domain size]
    domain_ = tf.cast(domain, tf.float32)
    p = p * domain_
    
    # clip for outliers (after advection)
    p = tf.clip_by_value(p, 0, domain_ - eps)

    # compute grid id
    cell_size = domain_ / tf.cast(res, tf.float32)
    # assert cell_size[0] == cell_size[1]
    cell_size = cell_size[0]
    indices_floor = tf.floor(p / cell_size)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)
    
    gc = (indices_floor+0.5)*cell_size # grid cell center
    r = gc - p # particle to grid center vector
    rr = []
    for n in range(-nsize,nsize+1):
        rr.append(r + n*cell_size)
    # rr = [r]

    # repulsive force with artificial weak spring [Ando and Tsuruno 2011]
    W_ = W(kernel)
    support_radius = radius*2 # particle spacing

    if is_2d:
        # [B,N,3], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 3])

        def interpolate_scatter2d(pos):
            dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1]
            q = tf.sqrt(dx**2 + dy**2) / support_radius

            disp = tf.stack([dy,dx], axis=-1)
            updates_raw = W_(q)
            updates_force = tf.expand_dims(updates_raw/q, axis=-1) * disp
            updates = tf.reshape(updates_force, [-1,2])

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize]])
            # indices_shift = tf.constant([[0] + [pos[0], pos[1]]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            img = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res,[2]), axis=-1))
            
            return img

        img = []
        for j in range(2*nsize+1):
            for i in range(2*nsize+1):
                vx = interpolate_scatter2d([j, i])
                img.append(vx)

        # vx, vx_rgb = interpolate_scatter2d([0, 0])
        # img.append(vx)
        # img_rgb.append(vx_rgb)

        img = tf.add_n(img)[:,::-1] # flip in y
        img *= -alpha*support_radius
        return img
    else:
        # [B,N,4], last three has its integer indices including batch id
        indices = tf.concat([batch_indices, indices_int], axis=2)
        indices = tf.reshape(indices, [-1, 4])

        def interpolate_scatter3d(pos):
            dz,dy,dx = rr[pos[0]][...,0], rr[pos[1]][...,1], rr[pos[2]][...,2]
            q = tf.sqrt(dx**2 + dy**2 + dz**2) / support_radius

            updates_raw = W_(q, support_radius, is_3d=True)
            updates = tf.reshape(updates_raw, [-1])

            indices_loc = indices
            indices_shift = tf.constant([[0] + [pos[0]-nsize, pos[1]-nsize, pos[2]-nsize]])
            num_updates = tf.shape(indices_loc)[0]
            indices_shift = tf.tile(indices_shift, [num_updates, 1])
            indices_loc = indices_loc + indices_shift

            vox = tf.scatter_nd(indices_loc, updates, tf.concat(([batch_size],res), axis=-1))
            vox = tf.expand_dims(vox, axis=-1)

            return vox

        vox = []
        for k in range(2*nsize+1):
            for j in range(2*nsize+1):
                for i in range(2*nsize+1):
                    vx = interpolate_scatter3d([k,j,i])
                    vox.append(vx)

        vox = tf.add_n(vox)[:,:,::-1] # flip in y
        return vox

def p2g_(p, res):
    # p.shape: [B,N,2]
    batch_size = p.shape[0]
    num_points = tf.shape(p)[1]

    indices_floor = tf.floor(p)
    indices_int = tf.cast(indices_floor, tf.int32)
    batch_indices = tf.range(0, batch_size, 1)
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, num_points])
    batch_indices = tf.expand_dims(batch_indices, -1)

    indices = tf.concat([batch_indices, indices_int], axis=2)
    indices = tf.reshape(indices, [-1, 4])

    r = p - indices_floor  # fractional part
    # rr = [1.0 - r, r]
    rr = [r, 1-r]
    W_ = W('cubic')

    def interpolate_scatter3d(pos):
        # updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        dx,dy,dz = rr[pos[0]][...,0], rr[pos[1]][...,1], rr[pos[2]][...,2]
        updates_raw = W_(tf.sqrt(dx**2 + dy**2 + dz**2) / np.sqrt(3)) # normalized distance
        updates = tf.reshape(updates_raw, [-1])

        indices_loc = indices
        indices_shift = tf.constant([[0] + pos])
        num_updates = tf.shape(indices_loc)[0]
        indices_shift = tf.tile(indices_shift, [num_updates, 1])
        indices_loc = indices_loc + indices_shift

        voxels = tf.scatter_nd(indices_loc, updates, [batch_size]+res)
        return voxels

    voxels = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx = interpolate_scatter3d([k, j, i])
                voxels.append(vx)

    voxels = tf.expand_dims(tf.add_n(voxels), axis=-1)
    voxels = voxels[:,:,::-1] # flip
    voxels = tf.transpose(voxels, [0, 3, 2, 1, 4])
    voxels /= tf.reduce_max(voxels) # TODO: remove for multiple frames
    return voxels

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
    img = tf.cast(np.arange(25).reshape(1, 5, 5, 1), tf.float32)
    identity_matrix = tf.cast([1, 0, 0, 0, 1, 0], tf.float32)
    zoom_in_matrix = identity_matrix * 0.5
    identity_warped = batch_affine_warp2d(img, identity_matrix)
    zoom_in_warped = batch_affine_warp2d(img, zoom_in_matrix)
    with tf.Session() as sess:
        print(sess.run(img[0, :, :, 0]))

        # # mgrid test
        # print(sess.run(batch_mgrid(2, 5, 4)))

        print(sess.run(identity_warped[0, :, :, 0]))
        print(sess.run(zoom_in_warped[0, :, :, 0]))