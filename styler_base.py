#############################################################
# MIT License, Copyright © 2020, ETH Zurich, Byungsoo Kim
#############################################################
import tensorflow as tf
import numpy as np
import os
from util import *
from transform import advect
import vgg

class StylerBase(object):
    def __init__(self, self_dict):
        # get arguments
        for arg in vars(self_dict):
            setattr(self, arg, getattr(self_dict,arg))

        # inception network setting
        self.model_path = os.path.join(self.data_dir, self.model_dir, self.network)
        if 'inception' in self.model_path:
            self.graph = tf.compat.v1.Graph()
            self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)
            with tf.io.gfile.GFile(self.model_path, 'rb') as f:
                self.graph_def = tf.compat.v1.GraphDef()
                self.graph_def.ParseFromString(f.read())

            # fix checkerboard artifacts: ksize should be divisible by the stride size
            # but it changes scale
            if self.pool1:
                for n in self.graph_def.node:
                    if 'conv2d0_pre_relu/conv' in n.name:
                        n.attr['strides'].list.i[1:3] = [1,1]

    def _plugin_to_loss_net(self, d):
        # resize rendering if needed 
        if not np.isclose(self.resize_scale, 1):
            h = tf.cast(tf.multiply(float(self.resize_scale), tf.cast(tf.shape(d)[1], tf.float32)), tf.int32)
            w = tf.cast(tf.multiply(float(self.resize_scale), tf.cast(tf.shape(d)[2], tf.float32)), tf.int32)
            d = tf.compat.v1.image.resize(d, (h,w), method=tf.image.ResizeMethod.BILINEAR) # upsample w/ BICUBIC -> artifacts

        # change the range of d image [0-1] to [0-255]
        d = d*255
        if not 'c' in self.target_field:
            d = tf.concat([d]*3, axis=-1) # [B,H,W,3]
        d = tf.reshape(d, [tf.shape(d)[0],tf.shape(d)[1],tf.shape(d)[2],3])
        self.d_img = d

        # plug-in to the pre-trained network
        if 'vgg' in self.model_path:
            self.sess = tf.compat.v1.InteractiveSession()
            self.layers = vgg.load_vgg(d, self.model_path, self.sess)
            print(self.layers.keys())
        else:
            # imagenet_mean = 117.0
            # d_preprocessed = d - vggimagenet_mean
            tf.import_graph_def(self.graph_def, {'input': vgg.preprocess(d)})
            self.layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
            print(self.layers)

    def _transport(self, g, v, a, b, recursive=True):
        # g: [H,W,1 or 2], v: [N,H,W,2]
        if a < b:
            if recursive:
                for i in range(a,b):
                    g = self.sess.run(self.adv, {self.g: g[None,:], self.u: v[i,None]})[0]
            else:
                # forward once
                g = self.sess.run(self.adv, {self.g: g[None,:], self.u: v[a,None]*(b-a)})[0]
        elif a > b:
            if recursive:
                for i in reversed(range(b,a)):
                    g = self.sess.run(self.adv, {self.g: g[None,:], self.u: -v[i,None]})[0]
            else:
                g = self.sess.run(self.adv, {self.g: g[None,:], self.u: -v[a-1,None]*(a-b)})[0]
        return g

    def _transport_tf(self, v, a, b, recursive=True):
        if a < b:
            if recursive:
                for i in range(a,b):
                    v = advect(v, tf.expand_dims(self.u[i], axis=0))
            else:
                v = advect(v, tf.expand_dims(self.u[a]*(b-a), axis=0))
        elif a > b:
            if recursive:
                for i in reversed(range(b,a)):
                    v = advect(v, tf.expand_dims(-self.u[i], axis=0))
            else:
                v = advect(v, tf.expand_dims(-self.u[a-1]*(a-b), axis=0))
        return v

    def _layer(self, layer):
        if 'input' in layer: return self.d_img
        if 'vgg' in self.model_path: return self.layers[layer]
        else: return self.graph.get_tensor_by_name("import/%s:0" % layer)

    def _gram_matrix(self, x):
        g_ = []
        for i in range(self.batch_size):
            F = tf.reshape(x[i], (-1, x.shape[-1]))
            g = tf.matmul(tf.transpose(F), F)
            g_.append(g)
        return tf.stack(g_, axis=0)

    def _hist_match(self, s, t, mask=None):
        m_ = []
        sm_ = []
        for i in range(self.batch_size):
            m_c = []
            sm_c = []
            for j in range(s.shape[-1]):
                s_ = s[i,...,j]
                if mask is not None:
                    nz = tf.not_equal(mask[i,...,0], 0)
                    s_ = tf.boolean_mask(s_, nz)
                    sm_c.append(s_)
                result = histogram_match_tf(s_, t[i,...,j])
                m_c.append(result['matched'])
            m_.append(tf.stack(m_c, axis=-1))
            if mask is not None:
                sm_.append(tf.stack(sm_c, axis=-1))
        if mask is not None:
            return m_, sm_
            # return tf.stack(m_, axis=0), tf.stack(sm_, axis=0)
        else:
            return tf.stack(m_, axis=0)

    def _loss(self, params):
        self.content_loss = 0
        self.style_loss = 0
        self.style_loss_layer = []
        self.hist_loss = 0
        self.hist_loss_layer = []
        self.total_loss = 0

        if self.w_content:
            feature = self._layer(self.content_layer) # assert only one layer
            if self.content_img is not None:
                self.content_feature = tf.compat.v1.placeholder(tf.float32, name='content_feature_%s' % self.content_layer)
                # self.content_loss -= tf.reduce_mean(feature*self.content_feature) # dot
                self.content_loss += tf.reduce_mean(tf.math.squared_difference(feature, 
                                               self.content_feature*self.w_content_amp))
            else:
                if self.content_channel:
                    self.content_loss -= tf.reduce_mean(feature[...,self.content_channel])
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,:self.content_channel]))
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,self.content_channel+1:]))
                else:
                    self.content_loss -= tf.reduce_mean(feature)

            self.total_loss += self.content_loss*self.w_content

        if self.w_style and self.style_img is not None:
            self.style_features = []
            for style_layer, w_style_layer in zip(self.style_layer, self.w_style_layer):
                feature = self._layer(style_layer)
                f_shp = tf.shape(feature)
                gram_denom = tf.cast(2*f_shp[1]*f_shp[2]*f_shp[3], tf.float32)
                
                style_feature = tf.compat.v1.placeholder(tf.float32, shape=feature.shape, name='style_feature_%s' % style_layer)
                # style_denom = tf.cast(2*f_shp[1]*f_shp[2]*f_shp[3], tf.float32)
                f_shp_ = tf.shape(style_feature)
                style_denom = tf.cast(2*f_shp_[1]*f_shp_[2]*f_shp_[3], tf.float32)
                self.style_features.append(style_feature)
                
                if self.style_mask:
                    style_mask = tf.compat.v1.image.resize(self.d_gray, (f_shp[1],f_shp[2]), method=tf.image.ResizeMethod.BICUBIC)
                    feature *= style_mask
                    area_mask = tf.reduce_sum(style_mask[...,0], axis=[1,2], keepdims=True)
                    gram_denom = 2*area_mask*tf.cast(f_shp[3], tf.float32)

                    if self.style_mask_on_ref:
                        style_feature *= style_mask
                        style_denom = 2*area_mask*tf.cast(f_shp[3], tf.float32)
                
                gram = self._gram_matrix(feature)
                gram /= gram_denom

                style_gram = self._gram_matrix(style_feature)
                style_gram /= style_denom

                style_loss = tf.reduce_sum(tf.math.squared_difference(gram, style_gram))
                self.style_loss_layer.append(style_loss)
                self.style_loss += w_style_layer*style_loss

            self.total_loss += self.style_loss*self.w_style

        if self.w_hist and self.style_img is not None:
            self.hist_features = []
            for hist_layer, w_hist_layer in zip(self.hist_layer, self.w_hist_layer):
                feature = self._layer(hist_layer)
                f_shp = tf.shape(feature)
                
                hist_feature = tf.compat.v1.placeholder(tf.float32, shape=feature.shape, name='hist_feature_%s' % hist_layer)
                self.hist_features.append(hist_feature)

                if self.style_mask:
                    hist_mask = tf.compat.v1.image.resize(self.d_gray, (f_shp[1],f_shp[2]), method=tf.image.ResizeMethod.BICUBIC)
                    matched_feature, feature_m = self._hist_match(feature, hist_feature, hist_mask)
                    hist_loss = 0
                    for m1, m2 in zip(matched_feature, feature_m):
                        hist_loss += tf.reduce_sum(tf.math.squared_difference(m1, m2))
                else:                        
                    matched_feature = self._hist_match(feature, style_feature)
                    hist_loss = tf.reduce_sum(tf.math.squared_difference(feature, matched_feature))
                    
                self.hist_loss_layer.append(hist_loss)
                self.hist_loss += w_style_layer*hist_loss
                        
            self.total_loss += self.hist_loss*self.w_hist

        if self.w_tv:
            self.tv_loss = tf.reduce_mean(tf.compat.v1.image.total_variation(self.d_img))
            self.total_loss += self.tv_loss*self.w_tv

        #######
        # loss for density preservation
        if self.w_density > 0:
            self.d_loss = 0
            self.d_pres = 0
            for i in range(self.batch_size):
                self.d_loss += tf.reduce_sum(self.d[i])**2
                self.d_pres += tf.reduce_sum(-tf.log(tf.abs(self.d[i]) + 1e-6))
            self.total_loss += (self.d_loss+self.d_pres*1e3)*self.w_density
        #######

        ######
        # loss for density correction
        if self.w_pressure > 0:
            self.pressure_loss = tf.reduce_mean(self.pressure**2)
            self.total_loss += self.pressure_loss*self.w_pressure
        ######

    def _content_feature(self, content_target, content_shp):
        if not np.isclose(self.resize_scale, 1):
            content_shp = [int(s*self.resize_scale) for s in content_shp]
        content_target_ = resize(content_target, content_shp, order=3) # bicubic for downsampling
        feature = self._layer(self.content_layer)
        feature_ = self.sess.run(feature, {self.d_img: [content_target_]*self.batch_size})

        if self.top_k > 0:
            assert('softmax2_pre_activation' in self.content_layer)
            feature_k_ = self.sess.run(tf.nn.top_k(np.abs(feature_), k=self.top_k))
            for i in range(len(feature_)):
                exclude_idx = np.setdiff1d(np.arange(feature_.shape[1]), feature_k_.indices[i])
                feature_[i,exclude_idx] = 0
        
        return feature_

    def _style_feature(self, style_target, style_shp=None):
        # mask for style texture
        style_m = None
        if style_target.shape[-1] == 4:
            style_m = style_target[...,-1]/255
            style_target = style_target[...,:-1]
            style_target *= np.stack([style_m]*3, axis=-1)
        
        if style_shp is not None:
            if not np.isclose(self.resize_scale, 1):
                style_shp = [int(s*self.resize_scale) for s in style_shp]
            style_target_ = resize(style_target, style_shp, order=3) # bicubic for downsampling
        else:
            style_target_ = style_target

        style_features = []
        for style_layer, w_style_layer in zip(self.style_layer, self.w_style_layer):
            style_feature = self._layer(style_layer)
            feed = {self.d_img: [style_target_]*self.batch_size}
            style_feature_ = self.sess.run(style_feature, feed)

            if style_m is not None:
                feature_mask_ = resize(style_m, style_feature_.shape[1:-1], order=3) # bicubic for downsampling
                feature_mask_ = np.stack([feature_mask_]*style_feature_.shape[-1], axis=-1)
                feature_mask_= np.stack([feature_mask_]*style_feature_.shape[0], axis=0)
                style_feature *= feature_mask_

            style_features.append(style_feature_)

        return style_features

    def _hist_feature(self, style_target, style_shp=None):
        # mask for style texture
        style_m = None
        if style_target.shape[-1] == 4:
            style_m = style_target[...,-1]/255
            style_target = style_target[...,:-1]
            style_target *= np.stack([style_m]*3, axis=-1)
        
        if style_shp is not None:
            if not np.isclose(self.resize_scale, 1):
                style_shp = [int(s*self.resize_scale) for s in style_shp]
            style_target_ = resize(style_target, style_shp, order=3) # bicubic for downsampling
        else:
            style_target_ = style_target

        hist_features = []
        for hist_layer, w_hist_layer in zip(self.hist_layer, self.w_hist_layer):
            hist_feature = self._layer(hist_layer)
            feed = {self.d_img: [style_target_]*self.batch_size}
            hist_feature_ = self.sess.run(hist_feature, feed)

            if style_m is not None:
                feature_mask_ = resize(style_m, hist_feature_.shape[1:-1], order=3) # bicubic for downsampling
                feature_mask_ = np.stack([feature_mask_]*hist_feature_.shape[-1], axis=-1)
                feature_mask_= np.stack([feature_mask_]*hist_feature_.shape[0], axis=0)
                hist_feature *= feature_mask_

            hist_features.append(hist_feature_)

        return hist_features

    def load_img(self, hw=None):
        self.content_img = None
        self.style_img = None
        
        if self.w_content > 0 and self.content_target:
            content_target = np.float32(Image.open(self.content_target))
            # remove alpha channel if exists
            if content_target.shape[-1] == 4:
                content_target = content_target[...,:-1]
            
            # crop
            if hw is not None:
                ratio = hw[1] / hw[0]
                content_target = crop_ratio(content_target, ratio)

            self.content_img = content_target

        if self.w_style > 0 and self.style_target:
            style_target = np.float32(Image.open(self.style_target))
            # print(style_target.shape)
            if self.style_tiling > 1:
                style_target = np.tile(style_target, (self.style_tiling, self.style_tiling, 1))
                # print(style_target.shape)
            
            # crop
            if hw is not None:
                ratio = hw[1] / hw[0]
                style_target = crop_ratio(style_target, ratio)

            # if style_target.shape[-1] == 4:
            #     style_m = style_target[...,-1]/255
            #     style_target = style_target[...,:-1]
            #     style_target *= np.stack([style_m]*3, axis=-1)
            #     # plt.imshow(style_target/255); plt.show()

            self.style_img = style_target