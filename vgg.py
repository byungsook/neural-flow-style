# http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
# https://github.com/singlasahil14/style-transfer/blob/master/nets/vgg.py
# https://medium.com/mlreview/getting-inception-architectures-to-work-with-style-transfer-767d53475bf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict
import os

slim = tf.contrib.slim

# _R_MEAN = 123.68
# _G_MEAN = 116.779
# _B_MEAN = 103.939
_R_MEAN = 0.485*255
_G_MEAN = 0.456*255
_B_MEAN = 0.406*255
_R_STD = 0.229*255
_G_STD = 0.224*255
_B_STD = 0.225*255


_content_layers_dict = {
  'vgg-16': ('conv2_2',), 
  'vgg-19': ('conv2_2',),
  'inception-v1': ('Conv2d_2c_3x3',),
  'inception-v2': ('Conv2d_2c_3x3',),
  'inception-v3': ('Conv2d_4a_3x3',),
  'inception-v4': ('Mixed_3a',),
  }

_style_layers_dict = {
  'vgg-16': ('conv3_1', 'conv4_1', 'conv5_1'), 
  'vgg-19': ('conv3_1', 'conv4_1', 'conv5_1'),
  'inception-v1': ('Conv2d_2c_3x3', 'Mixed_3c', 'Mixed_4b', 'Mixed_5b'),
  'inception-v2': ('Conv2d_2c_3x3', 'Mixed_3b', 'Mixed_4a', 'Mixed_5a'),
  'inception-v3': ('Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_6a', 'Mixed_7a'),
  'inception-v4': ('Mixed_4a', 'Mixed_5a', 'Mixed_6a', 'Mixed_7a'),
  }

def vgg_arg_scope(padding='SAME'):
  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding=padding) as arg_sc:
      return arg_sc

def preprocess(images):
  images -= tf.constant([ _R_MEAN ,  _G_MEAN,  _B_MEAN])
  # images /= tf.constant([ _R_STD ,  _G_STD,  _B_STD])
  return images

def repeat(inputs, repetitions, layer, *args, **kwargs):
  scope = kwargs.pop('scope', 'conv')
  end_points = kwargs.pop('end_points', OrderedDict())
  with tf.compat.v1.variable_scope(scope, 'Repeat', [inputs]):
    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs
    for i in range(repetitions):
      scope_name = scope + '_' + str(i+1)
      kwargs['scope'] = scope_name
      outputs = layer(outputs, *args, **kwargs)
      end_points[scope_name] = outputs
    return outputs, end_points

def vgg_16(inputs, scope='vgg_16', reuse=False, pool_fn=slim.avg_pool2d):
  with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    # Collect outputs for conv2d and pool_fn.
    with slim.arg_scope([slim.conv2d, pool_fn]):
      net, end_points = repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = pool_fn(net, [2, 2], scope='pool1')
      end_points['pool1'] = net
      net, end_points = repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool2')
      end_points['pool2'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool3')
      end_points['pool3'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool4')
      end_points['pool4'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool5')
      end_points['pool5'] = net
      return end_points

def vgg_19(inputs, scope='vgg_19', reuse=False, pool_fn=slim.avg_pool2d):
  with tf.compat.v1.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    # Collect outputs for conv2d and pool_fn.
    with slim.arg_scope([slim.conv2d, pool_fn]):
      net, end_points = repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = pool_fn(net, [2, 2], scope='pool1')
      end_points['pool1'] = net
      net, end_points = repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool2')
      end_points['pool2'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool3')
      end_points['pool3'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool4')
      end_points['pool4'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool5')
      end_points['pool5'] = net
      return end_points

def load_vgg(d, model_path, sess, pool_fn=slim.avg_pool2d):
    model_name = os.path.basename(model_path).split('.')[0] # vgg_16
    # print(model_name)
    vgg_in = preprocess(d)
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        if '16' in model_name: layers = vgg_16(vgg_in, pool_fn=pool_fn)
        else: layers = vgg_19(vgg_in, pool_fn=pool_fn)
            
    init = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables(model_name))
    init(sess)
    return layers