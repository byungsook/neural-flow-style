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
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def denoise(img, sigma):
    if sigma > 0:
        return gaussian_filter(img, sigma=sigma)
    else:
        return img

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

def resize(img, size=None, f=None, order=3):
    vmin, vmax = img.min(), img.max()
    if vmin < -1 or vmax > 1:
        img = (img - vmin) / (vmax-vmin) # [0,1]
    if size is not None:
        if img.ndim == 4:
            if len(size) == 4: size = size[:-1]
            img_ = []
            for i in range(img.shape[-1]):
                img_.append(skimage.transform.resize(img[...,i], size, order=order).astype(np.float32))
            img = np.stack(img_, axis=-1)
        elif img.ndim < 4:
            img = skimage.transform.resize(img, size, order=order).astype(np.float32)
        else:
            assert False
    else:
        img = skimage.transform.rescale(img, f, order=order).astype(np.float32)
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

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs_and_logger(config):
    # print(__file__)
    os.chdir(os.path.dirname(__file__))

    model_name = "{}_{}".format(get_time(), config.tag)
    config.log_dir = os.path.join(config.log_dir, model_name)
    
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
    filename = os.path.join(imgdir, '..', filename+'.mp4')
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