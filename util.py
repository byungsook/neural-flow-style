#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
from datetime import datetime
import os
import imageio
from glob import glob
import shutil
import numpy as np
from PIL import Image
import logging
import json
from scipy.ndimage import gaussian_filter, zoom
import skimage.transform
from functools import partial
import tensorflow as tf
import matplotlib.colors
import matplotlib.pyplot as plt
try:
    import open3d as o3d
except ImportError:    
    pass # leonhard
from matplotlib import cm
from subprocess import call
import sys

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = {
    1: k[:,:,None,None]/k.sum(),
    2: k[:,:,None,None]/k.sum()*np.eye(2, dtype=np.float32),
    3: k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32),
    4: k[:,:,None,None]/k.sum()*np.eye(4, dtype=np.float32),
    6: k[:,:,None,None]/k.sum()*np.eye(6, dtype=np.float32),
}
k_ = []
k2_ = [1,16**(1/3),36**(1/3),16**(1/3),1]
k2 = np.float32([1,16**(1/3),36**(1/3),16**(1/3),1])
k2 = np.outer(k2, k2)
for i in k2_:
    k_.append(k2*i)
k_ = np.floor(np.array(k_))
k5x5x5 = {
    1: k_[:,:,:,None,None]/k_.sum(),
    3: k_[:,:,:,None,None]/k_.sum()*np.eye(3, dtype=np.float32),
    5: k_[:,:,:,None,None]/k_.sum()*np.eye(5, dtype=np.float32),
}


def cosine_decay(global_step, decay_steps, learning_rate, factor):
    global_step = min(global_step, decay_steps)
    cos_decay = np.cos(np.pi * global_step / decay_steps) # [1, -1]
    cos_decay = (cos_decay + 1)*0.5*(factor-1) + 1 # [factor, 1]
    return learning_rate * cos_decay # 2lr -> lr


def lap_split(img, is_3d, k):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        if is_3d:
            lo = tf.nn.conv3d(img, k, [1,2,2,2,1], 'SAME')
            lo2 = tf.nn.conv3d_transpose(lo, k*5, tf.shape(img), [1,2,2,2,1])
        else:
            lo = tf.nn.conv2d(img, k, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n, is_3d, k):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img, is_3d, k)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels, is_3d, k):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            if is_3d:
                img = tf.nn.conv3d_transpose(img, k*5, tf.shape(hi), [1,2,2,2,1]) + hi
            else:
                img = tf.nn.conv2d_transpose(img, k*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=3, is_3d=False, c=1):
    '''Perform the Laplacian pyramid normalization.'''
    if scale_n == 0:
        m = tf.reduce_mean(tf.abs(img))
        return img/tf.maximum(m, 1e-7)
    else:
        if is_3d:
            k = k5x5x5[c]
        else:
            k = k5x5[c]

        img = tf.expand_dims(img, 0)
        tlevels = lap_split_n(img, scale_n, is_3d, k)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels, is_3d, k)
        return out[0]

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.compat.v1.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_tf(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, is_3d=False):
    if is_3d:
        # b, d, h, w, c = int_shape(x)
        shp = tf.shape(x)
        b, d, h, w, c = shp[0], shp[1], shp[2], shp[3], shp[4]
        hw = tf.reshape(tf.transpose(x, [0,2,3,1,4]), [b,h,w,d*c])
        h, w = size[1], size[2]
        hw = tf.compat.v1.image.resize(hw, (h,w), method=method)
        hw = tf.reshape(hw, [b,h,w,d,c])
        dh = tf.reshape(tf.transpose(hw, [0,3,1,2,4]), [b,d,h,w*c])
        d = size[0]
        dh = tf.compat.v1.image.resize(dh, (d,h), method=method)
        x = tf.reshape(dh, [b,d,h,w,c])
    else:
        x = tf.compat.v1.image.resize(x, size, method=method)
    return x

def rescale_tf(x, scale, method=tf.image.ResizeMethod.BILINEAR, is_3d=False):
    if is_3d:
        # b, d, h, w, c = int_shape(x)
        shp = tf.shape(x)
        b, d, h, w, c = shp[0], shp[1], shp[2], shp[3], shp[4]
        hw = tf.reshape(tf.transpose(x, [0,2,3,1,4]), [b,h,w,d*c])
        h = tf.cast(tf.cast(h, tf.float32)*scale, tf.int32)
        w = tf.cast(tf.cast(w, tf.float32)*scale, tf.int32)

        hw = tf.compat.v1.image.resize(hw, (h,w), method=method)
        hw = tf.reshape(hw, [b,h,w,d,c])
        dh = tf.reshape(tf.transpose(hw, [0,3,1,2,4]), [b,d,h,w*c])
        d = tf.cast(tf.cast(d, tf.float32)*scale, tf.int32)
        dh = tf.compat.v1.image.resize(dh, (d,h), method=method)
        x = tf.reshape(dh, [b,d,h,w,c])
    else:
        # b, h, w, c = int_shape(x)
        shp = tf.shape(x)
        b, h, w, c = shp[0], shp[1], shp[2], shp[3]
        h = tf.cast(tf.cast(h, tf.float32)*scale, tf.int32)
        w = tf.cast(tf.cast(w, tf.float32)*scale, tf.int32)
        x = tf.compat.v1.image.resize(x, (h,w), method=method)
    return x

def denoise(img, sigma):
    return gaussian_filter(img, sigma=sigma)
    # if sigma > 0:
    #     return gaussian_filter(img, sigma=sigma)
    # else:
    #     return img

def crop_ratio(img, ratio):
    hw_t = img.shape[:2]
    ratio_t = hw_t[1] / float(hw_t[0])
    if ratio_t > ratio:
        hw_ = [hw_t[0], int(hw_t[0]*ratio)]
    else:
        hw_ = [int(hw_t[1]/ratio), hw_t[1]]
    assert(hw_[0] <= hw_t[0] and hw_[1] <= hw_t[1])
    o = [int((hw_t[0]-hw_[0])*0.5), int((hw_t[1]-hw_[1])*0.5)]
    return img[o[0]:o[0]+hw_[0], o[1]:o[1]+hw_[1]]

def resize(img, size=None, f=None, order=1):
    vmin, vmax = img.min(), img.max()
    if vmin < -1 or vmax > 1:
        img = (img - vmin) / (vmax-vmin) # [0,1]
    if size is not None:
        if img.ndim == 4:
            if len(size) == 4: size = size[:-1]
            img_ = []
            for i in range(img.shape[-1]):
                img_.append(skimage.transform.resize(img[...,i], size, order=order, mode='constant', anti_aliasing=True).astype(np.float32))
            img = np.stack(img_, axis=-1)
        elif img.ndim < 4:
            img = skimage.transform.resize(img, size, order=order, mode='constant', anti_aliasing=True).astype(np.float32)
        else:
            assert False
    else:
        img = skimage.transform.rescale(img, f, order=order, multichannel=None, mode='constant', anti_aliasing=True).astype(np.float32)
    if vmin < -1 or vmax > 1:
        return img * (vmax-vmin) + vmin
    else:
        return img

def save_density(d, d_path):
    im = d*255
    im = np.stack((im,im,im), axis=-1).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(d_path)

def yuv2rgb(y,u,v):
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/image_ops_impl.py
    r = y + 1.13988303*v
    g = y - 0.394642334*u - 0.58062185*v
    b = y + 2.03206185*u
    # r = y + 1.4746*v
    # g = y - 0.16455*u - 0.57135*v
    # b = y + 1.8814*u
    # ## JPEG
    # r = y + 1.402*v
    # g = y - 0.344136*u - 0.714136*v
    # b = y + 1.772*u
    return r,g,b

def rgb2yuv(r,g,b):
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14714119*r - 0.28886916*g + 0.43601035*b
    v = 0.61497538*r - 0.51496512*g - 0.10001026*b
    return y,u,v

def hsv2rgb(h,s,v):
    c = s * v
    m = v - c
    dh = h * 6
    h_category = tf.cast(dh, tf.int32)
    fmodu = tf.mod(dh, 2)
    x = c * (1 - tf.abs(fmodu - 1))
    component_shape = tf.shape(h)
    dtype = h.dtype
    rr = tf.zeros(component_shape, dtype=dtype)
    gg = tf.zeros(component_shape, dtype=dtype)
    bb = tf.zeros(component_shape, dtype=dtype)
    h0 = tf.equal(h_category, 0)
    rr = tf.where(h0, c, rr)
    gg = tf.where(h0, x, gg)
    h1 = tf.equal(h_category, 1)
    rr = tf.where(h1, x, rr)
    gg = tf.where(h1, c, gg)
    h2 = tf.equal(h_category, 2)
    gg = tf.where(h2, c, gg)
    bb = tf.where(h2, x, bb)
    h3 = tf.equal(h_category, 3)
    gg = tf.where(h3, x, gg)
    bb = tf.where(h3, c, bb)
    h4 = tf.equal(h_category, 4)
    rr = tf.where(h4, x, rr)
    bb = tf.where(h4, c, bb)
    h5 = tf.equal(h_category, 5)
    rr = tf.where(h5, c, rr)
    bb = tf.where(h5, x, bb)
    r = rr + m
    g = gg + m
    b = bb + m
    return r,g,b

# Util function to match histograms
def match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image (source to template)

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # plt.figure()
    # plt.plot(range(len(s_quantiles)), s_quantiles, range(len(t_quantiles)), t_quantiles)
    # plt.show()

    return interp_t_values[bin_idx].reshape(oldshape)

def histogram_match_tf(source, template, hist_bins=255):
    shape = tf.shape(source)

    source = tf.layers.flatten(source)
    template = tf.layers.flatten(template)

    # get the set of unique pixel values and their corresponding indices and counts
    # hist_bins = trainer.hist_bins

    # Defining the 'x_axis' of the histogram
    max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
    min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])

    hist_delta = (max_value - min_value)/hist_bins

    # Getting the x-axis for each value
    hist_range = tf.range(min_value, max_value, hist_delta)
    # I don't want the bin values; instead, I want the average value of each bin, which is 
    # lower_value + hist_delta/2
    hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))

    # Now, making fixed width histograms on this hist_axis
    s_hist = tf.histogram_fixed_width(source, 
                                    [min_value, max_value],
                                    nbins=hist_bins, 
                                    dtype=tf.int64
                                    )

    t_hist = tf.histogram_fixed_width(template, 
                                    [min_value, max_value],
                                    nbins=hist_bins, 
                                    dtype=tf.int64
                                    )

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = tf.cumsum(s_hist)
    s_quantiles /= s_quantiles[-1]
    
    t_quantiles = tf.cumsum(t_hist)
    t_quantiles /= t_quantiles[-1]

    from scipy.interpolate import interp1d
    def intp(x, xp):
        intp = interp1d(xp, np.arange(hist_bins), bounds_error=False, fill_value=(0,hist_bins-1))
        return np.round(intp(x)).astype(np.int64)
    nearest_indices = tf.py_func(intp, [s_quantiles, t_quantiles], tf.int64)

    # nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), 
    #                             s_quantiles, dtype=tf.int64)


    # Finding the correct s-histogram bin for every element in source
    s_bin_index = tf.cast((source-min_value)/hist_delta, tf.int64)

    ## In the case where an activation function of 0-1 is used, then there might be some index exception errors. 
    ## This is to deal with those
    s_bin_index = tf.clip_by_value(s_bin_index, 0, hist_bins-1)

    # Matching it to the correct t-histogram bin, and then making it the correct shape again
    matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
    matched = tf.reshape(matched_to_t, shape)

    # to compare histograms
    m_hist = tf.histogram_fixed_width(matched_to_t,
                                    [min_value, max_value],
                                    nbins=hist_bins, 
                                    dtype=tf.int64
                                    )

    # self.s_hist = s_hist
    # self.t_hist = t_hist
    # self.m_hist = m_hist
    # self.s_quantiles = s_quantiles
    # self.t_quantiles = t_quantiles
    result = {
        's_hist': s_hist,
        't_hist': t_hist,
        'm_hist': m_hist,
        'matched': matched,
    }
    return result

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs_and_logger(config):
    config.command = str(sys.argv)

    # print(__file__)
    os.chdir(os.path.dirname(__file__))

    model_name = "{}_{}".format(get_time(), config.tag)
    config.log_dir = os.path.join(config.log_dir, config.dataset, model_name)
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    save_config(config)

def save_config(config):
    param_path = os.path.join(config.log_dir, "params.json")

    print("[*] MODEL dir: %s" % config.log_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_video(imgdir, filename, ext='png', fps=24, delete_imgdir=False):
    filename = os.path.join(imgdir, filename+'.mp4')
    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)

    imgs = glob("{}/*.{}".format(imgdir, ext))
    imgs = sorted(imgs, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # print(imgs)
    for img in imgs:
        im = imageio.imread(img)
        writer.append_data(im)
    
    writer.close()
    
    if delete_imgdir: shutil.rmtree(imgdir)

def v2rgb(v):
    # lazyfluid colormap    
    theta = np.arctan2(-v[...,0], -v[...,1])
    theta = (theta + np.pi) / (2*np.pi)
    r = np.sqrt(v[...,0]**2+v[...,1]**2)
    r_max = r.max()
    r /= r_max
    o = np.ones_like(r)
    hsv = np.stack((theta,r,o), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb = (rgb*255).astype(np.uint8)
    return rgb

# v_path = 'E:/neural-flow-style\log\smoke_plume_f200/1104_165443_test/0_0_151_v.npz'
# v_sty = np.load(v_path)['v']
# # v_path = 'D:\dev\deep-fluids\data\smoke3_vel5_buo3_f250/v/0_0_150.npz'
# # v_sty = np.load(v_path)['x'][::-1,:,16]
# # import matplotlib.pyplot as plt
# # plt.figure()
# # plt.subplot(131)
# # plt.imshow(v_sty[...,0])
# # plt.subplot(132)
# # plt.imshow(v_sty[...,1])
# # plt.subplot(133)
# # plt.imshow(v_sty[...,2])
# # plt.show()
# # v_sty = np.stack((v_sty[...,1], v_sty[...,0]), axis=-1)
# # v_path = 'E:/neural-flow-style\log\smoke_plume_f200/1104_165443_test/test.npz'
# im = Image.fromarray(v2rgb(v_sty))
# d_file_path = v_path[:-4]+'.png'
# im.save(d_file_path)

# save_video('E:/neural-flow-style\log\smoke_plume_f200/1102_064036_adv_s4_w0.05_volc', 'E:/neural-flow-style\log\smoke_plume_f200/1102_064036_adv_s4_w0.05_volc')

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False, gray=True):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    if padding == 0:
        if gray:
            grid = np.zeros([height * ymaps, width * xmaps], dtype=np.uint8)
        else:
            grid = np.zeros([height * ymaps, width * xmaps, 3], dtype=np.uint8)
    else:
        if gray:
            grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2], dtype=np.uint8)
        else:
            grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if padding == 0:
                h, h_width = y * height, height
                w, w_width = x * width, width
            else:
                h, h_width = y * height + 1 + padding // 2, height - padding
                w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False, single=False, gray=True):
    if not single:
        ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                          normalize=normalize, scale_each=scale_each, gray=gray)
    else:
        # h, w = tensor.shape[0], tensor.shape[1]
        # if gray:
        #     ndarr = np.zeros([h,w], dtype=np.uint8)
        # else:
        #     ndarr = np.zeros([h,w,3], dtype=np.uint8)
        ndarr = tensor
        
    im = Image.fromarray(ndarr)
    im.save(filename)

def draw_voxel(d):
    vox = np.argwhere(d>0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vox)
    c = d[d>0]
    c = np.stack([c]*3, axis=-1)
    pcd.colors = o3d.utility.Vector3dVector(c)
    # vol = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
    o3d.visualization.draw_geometries([pcd])

def npz2vdb(d, vdb_exe, d_path):
    np.savez_compressed(d_path, x=d[:,::-1])
    sh = [vdb_exe, 'npz2vdb.py', '--src_path='+d_path]
    call(sh, shell=True)

def draw_pt(pt, pv=None, pc=None, dt=1, is_2d=True, bbox=None):
    geom = []
    
    # # bounding box
    # xmin, xmax = 0, 0
    # ymin, ymax = 0, 0
    # zmin, zmax = -1, 1
    # for i, p in enumerate(pt):
    #     if i == 0:
    #         xmin, xmax = p[:,0].min(), p[:,0].max()
    #         ymin, ymax = p[:,1].min(), p[:,1].max()
    #         if not is_2d:
    #             zmin, zmax = p[:,2].min(), p[:,2].max()
    #     else:
    #         xmin = min(p[:,0].min(), xmin)
    #         xmax = max(p[:,0].max(), xmax)
    #         ymin = min(p[:,1].min(), ymin)
    #         ymax = max(p[:,1].max(), ymax)
    #         if not is_2d:
    #             zmin = min(p[:,2].min(), zmin)
    #             zmax = max(p[:,2].max(), zmax)

    # bbox = [
    #     [xmin, ymin, zmin],
    #     [xmax, ymax, zmax]
    #     ]
    if bbox is not None:
        bp = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    bp.append([bbox[i][0],bbox[j][1],bbox[k][2]])
        bl = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]
        bbox_line = o3d.geometry.LineSet()
        bbox_line.points = o3d.utility.Vector3dVector(bp)
        bbox_line.lines = o3d.utility.Vector2iVector(bl)
        geom.append(bbox_line)

    # gizmo
    # gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(
    gizmo = o3d.geometry.create_mesh_coordinate_frame(
        size=1, origin=[0, 0, 0])
    geom.append(gizmo)

    # particles
    pcd = o3d.geometry.PointCloud()
    geom.append(pcd)
    # pcd_idx = len(geom)
    # for i in range(pt.shape[1]):
    #     mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    #     geom.append(mesh_sphere)
    #     if i == 20: break

    # velocity
    if pv is not None:
        line_set = o3d.geometry.LineSet()
        geom.append(line_set)

    # draw_pt.vol = None

    draw_pt.t = 0
    def loadframe(vis):
        print('frame', draw_pt.t)
        p = pt[draw_pt.t]
        if is_2d:
            pz = np.zeros([p.shape[0],1])
            p = np.concatenate((p,pz), axis=-1)

        pcd.points = o3d.utility.Vector3dVector(p)

        # if draw_pt.vol is not None: vis.remove_geometry(draw_pt.vol)
        # draw_pt.vol = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
        # vis.add_geometry(draw_pt.vol)

        # for i in range(p.shape[0]):
        #     geom[pcd_idx+i].translate(translation=p[i])
        #     geom[pcd_idx+i].compute_vertex_normals()
        #     if i == 20: break

        if pc is not None:
            c = pc[draw_pt.t]
            pcd.colors = o3d.utility.Vector3dVector(c)

        if pv is not None:
            v = pv[draw_pt.t]
            if is_2d:
                vz = np.zeros([v.shape[0],1])
                v = np.concatenate((v,vz), axis=-1)

            p_ = p + v*dt
            p = np.concatenate((p,p_), axis=0)
            l0 = np.arange(v.shape[0])
            l1 = np.arange(v.shape[0],2*v.shape[0])
            l = np.stack((l0,l1), axis=-1)
        
            if pc is None:        
                c = np.sqrt(np.sum(v**2, axis=-1))
                c /= c.max()
                c = cm.Blues(1 - c)[...,:-1]
                pcd.colors = o3d.utility.Vector3dVector(c)

            # for i in range(c.shape[0]):
            #     geom[pcd_idx+i].paint_uniform_color(c[i])
            #     if i == 20: break

            line_set.points = o3d.utility.Vector3dVector(p)
            line_set.lines = o3d.utility.Vector2iVector(l)
            line_set.colors = o3d.utility.Vector3dVector(c)
        
        elif pc is None:
            c = np.zeros_like(p)
            # c[...,-1] = 0.8 # 
            pcd.colors = o3d.utility.Vector3dVector(c)

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        # cam = vis.get_view_control()
        # if cam is not None:
        #     param = cam.convert_to_pinhole_camera_parameters()
        #     print('intrisic', param.intrinsic.width, param.intrinsic.height)
        #     print(param.intrinsic.intrinsic_matrix)
        #     print('extrisic\n', param.extrinsic)

        # p = pt[draw_pt.t]
        # for i in range(p.shape[0]):
        #     geom[pcd_idx+i].translate(translation=-p[i])
        #     if i == 20: break

    vis = o3d.visualization.Visualizer()
    loadframe(vis) # for the first frame

    def nextframe(vis):
        # print('nextframe')
        draw_pt.t += 1
        if draw_pt.t == len(pt):
            draw_pt.t = 0
        loadframe(vis)
        return False
        
    def prevframe(vis):
        # print('prevframe')
        draw_pt.t -= 1
        if draw_pt.t == -1:
            draw_pt.t = len(pt)-1
        loadframe(vis)
        return False

    key_to_callback = {}
    key_to_callback[ord(",")] = prevframe
    key_to_callback[ord(".")] = nextframe
    o3d.visualization.draw_geometries_with_key_callbacks(geom, key_to_callback)