# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import LandmarkImage,LandmarkImage_98


def mobilenet_v2(input, weight_decay, batch_norm_params):
    features = {}
    with tf.variable_scope('Mobilenet'):
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                            activation_fn=tf.nn.relu6,\
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('Mobilnet input shape({}): {}'.format(input.name, input.get_shape()))

            # 96*96*3   112*112*3
            conv_1 = slim.convolution2d(input, 32, [3, 3], stride=2, scope='conv_1')
            print(conv_1.name, conv_1.get_shape())

            # 48*48*32  56*56*32
            conv2_1 = slim.separable_convolution2d(conv_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv2_1/dwise')
            print(conv2_1.name, conv2_1.get_shape())
            conv2_1 = slim.convolution2d(conv2_1, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv2_1/linear')
            print(conv2_1.name, conv2_1.get_shape())
            features['feature2'] = conv2_1
            # 48*48*16  56*56*16
            conv3_1 = slim.convolution2d(conv2_1, 96, [1, 1], stride=1, scope='conv3_1/expand')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.separable_convolution2d(conv3_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_1/dwise')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.convolution2d(conv3_1, 24, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_1/linear')
            print(conv3_1.name, conv3_1.get_shape())

            conv3_2 = slim.convolution2d(conv3_1, 144, [1, 1], stride=1, scope='conv3_2/expand')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.separable_convolution2d(conv3_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_2/dwise')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.convolution2d(conv3_2, 24, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_2/linear')
            print(conv3_2.name, conv3_2.get_shape())
            block_3_2 = conv3_1 + conv3_2
            print(block_3_2.name, block_3_2.get_shape())

            features['feature3'] = block_3_2
            features['pfld'] = block_3_2
            # 24*24*24   28*28*24
            conv4_1 = slim.convolution2d(block_3_2, 144, [1, 1], stride=1, scope='conv4_1/expand')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.separable_convolution2d(conv4_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_1/dwise')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.convolution2d(conv4_1, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_1/linear')
            print(conv4_1.name, conv4_1.get_shape())

            conv4_2 = slim.convolution2d(conv4_1, 192, [1, 1], stride=1, scope='conv4_2/expand')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.separable_convolution2d(conv4_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_2/dwise')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.convolution2d(conv4_2, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_2/linear')
            print(conv4_2.name, conv4_2.get_shape())
            block_4_2 = conv4_1 + conv4_2
            print(block_4_2.name, block_4_2.get_shape())

            conv4_3 = slim.convolution2d(block_4_2, 192, [1, 1], stride=1, scope='conv4_3/expand')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.separable_convolution2d(conv4_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_3/dwise')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.convolution2d(conv4_3, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_3/linear')
            print(conv4_3.name, conv4_3.get_shape())
            block_4_3 = block_4_2 + conv4_3
            print(block_4_3.name, block_4_3.get_shape())

            # 12*12*32   14*14*32
            features['feature4'] = block_4_3
            conv5_1 = slim.convolution2d(block_4_3, 192, [1, 1], stride=1, scope='conv5_1/expand')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.separable_convolution2d(conv5_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_1/dwise')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.convolution2d(conv5_1, 64, [1, 1], stride=1,activation_fn=None,
                                         scope='conv5_1/linear')
            print(conv5_1.name, conv5_1.get_shape())

            conv5_2 = slim.convolution2d(conv5_1, 384, [1, 1], stride=1, scope='conv5_2/expand')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.separable_convolution2d(conv5_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_2/dwise')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.convolution2d(conv5_2, 64, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_2/linear')
            print(conv5_2.name, conv5_2.get_shape())
            block_5_2 = conv5_1 + conv5_2
            print(block_5_2.name, block_5_2.get_shape())

            conv5_3 = slim.convolution2d(block_5_2, 384, [1, 1], stride=1, scope='conv5_3/expand')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.separable_convolution2d(conv5_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_3/dwise')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.convolution2d(conv5_3, 64, [1, 1], stride=1,  activation_fn=None,
                                         scope='conv5_3/linear')
            print(conv5_3.name, conv5_3.get_shape())
            block_5_3 = block_5_2 + conv5_3
            print(block_5_3.name, block_5_3.get_shape())

            conv5_4 = slim.convolution2d(block_5_3, 384, [1, 1], stride=1, scope='conv5_4/expand')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.separable_convolution2d(conv5_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_4/dwise')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.convolution2d(conv5_4, 64, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_4/linear')
            print(conv5_4.name, conv5_4.get_shape())
            block_5_4 = block_5_3 + conv5_4
            print(block_5_4.name, block_5_4.get_shape())

            # 6*6*64    7*7*64
            conv6_1 = slim.convolution2d(block_5_4, 384, [1, 1], stride=1, scope='conv6_1/expand')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.separable_convolution2d(conv6_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_1/dwise')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.convolution2d(conv6_1, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_1/linear')
            print(conv6_1.name, conv6_1.get_shape())

            conv6_2 = slim.convolution2d(conv6_1, 576, [1, 1], stride=1, scope='conv6_2/expand')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.separable_convolution2d(conv6_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_2/dwise')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.convolution2d(conv6_2, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_2/linear')
            print(conv6_2.name, conv6_2.get_shape())
            block_6_2 = conv6_1 + conv6_2
            print(block_6_2.name, block_6_2.get_shape())

            conv6_3 = slim.convolution2d(block_6_2, 576, [1, 1], stride=1, scope='conv6_3/expand')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.separable_convolution2d(conv6_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_3/dwise')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.convolution2d(conv6_3, 96, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_3/linear')
            print(conv6_3.name, conv6_3.get_shape())
            block_6_3 = block_6_2 + conv6_3
            print(block_6_3.name, block_6_3.get_shape())

            features['feature5'] = block_6_3
            # 6*6*96    7*7*96
            conv7_1 = slim.convolution2d(block_6_3, 576, [1, 1], stride=1, scope='conv7_1/expand')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.separable_convolution2d(conv7_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_1/dwise')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.convolution2d(conv7_1, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_1/linear')
            print(conv7_1.name, conv7_1.get_shape())

            conv7_2 = slim.convolution2d(conv7_1, 960, [1, 1], stride=1, scope='conv7_2/expand')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.separable_convolution2d(conv7_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_2/dwise')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.convolution2d(conv7_2, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_2/linear')
            print(conv7_2.name, conv7_2.get_shape())
            block_7_2 = conv7_1 + conv7_2
            print(block_7_2.name, block_7_2.get_shape())


            conv7_3 = slim.convolution2d(block_7_2, 960, [1, 1], stride=1, scope='conv7_3/expand')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.separable_convolution2d(conv7_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_3/dwise')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.convolution2d(conv7_3, 160, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_3/linear')
            print(conv7_3.name, conv7_3.get_shape())
            block_7_3 = block_7_2 + conv7_3
            print(block_7_3.name, block_7_3.get_shape())

            conv7_4 = slim.convolution2d(block_7_3, 960, [1, 1], stride=1, scope='conv7_4/expand')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.separable_convolution2d(conv7_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_4/dwise')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.convolution2d(conv7_4, 320, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_4/linear')
            print(conv7_4.name, conv7_4.get_shape())
            features['feature6'] = conv7_4
    return features

def pfld_inference(input, weight_decay, batch_norm_params):

    coefficient = 1
    with tf.variable_scope('pfld_inference'):
        features = {}
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                            activation_fn=tf.nn.relu6,\
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            print('PFLD input shape({}): {}'.format(input.name, input.get_shape()))
            # 112*112*3
            conv1 = slim.convolution2d(input, 64*coefficient, [3, 3], stride=2, scope='conv_1')
            print(conv1.name, conv1.get_shape())

            # 56*56*64
            conv2 = slim.separable_convolution2d(conv1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv2/dwise')
            print(conv2.name, conv2.get_shape())

            # 56*56*64
            conv3_1 = slim.convolution2d(conv2, 128, [1, 1], stride=2, scope='conv3_1/expand')
            print(conv3_1.name,conv3_1.get_shape())
            conv3_1 = slim.separable_convolution2d(conv3_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_1/dwise')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.convolution2d(conv3_1,64*coefficient,[1,1],stride=1,activation_fn=None,scope='conv3_1/linear')
            print(conv3_1.name, conv3_1.get_shape())

            conv3_2 = slim.convolution2d(conv3_1, 128, [1, 1], stride=1,scope='conv3_2/expand')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.separable_convolution2d(conv3_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_2/dwise')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.convolution2d(conv3_2, 64*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv3_2/linear')
            print(conv3_2.name, conv3_2.get_shape())

            block3_2 = conv3_1 + conv3_2
            print(block3_2.name,block3_2.get_shape())

            conv3_3 = slim.convolution2d(block3_2, 128, [1, 1], stride=1, scope='conv3_3/expand')
            print(conv3_3.name, conv3_3.get_shape())
            conv3_3 = slim.separable_convolution2d(conv3_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_3/dwise')
            print(conv3_3.name, conv3_3.get_shape())
            conv3_3 = slim.convolution2d(conv3_3, 64*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv3_3linear')
            print(conv3_3.name, conv3_3.get_shape())

            block3_3 = block3_2 + conv3_3
            print(block3_3.name,block3_3.get_shape())

            conv3_4 = slim.convolution2d(block3_3, 128, [1, 1], stride=1, scope='conv3_4/expand')
            print(conv3_4.name, conv3_4.get_shape())
            conv3_4 = slim.separable_convolution2d(conv3_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_4/dwise')
            print(conv3_4.name, conv3_4.get_shape())
            conv3_4 = slim.convolution2d(conv3_4, 64*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv3_4/linear')
            print(conv3_4.name, conv3_4.get_shape())

            block3_4 = block3_3 + conv3_4
            print(block3_4.name,block3_4.get_shape())

            conv3_5 = slim.convolution2d(block3_4, 128, [1, 1], stride=1,scope='conv3_5/expand')
            print(conv3_5.name, conv3_5.get_shape())
            conv3_5 = slim.separable_convolution2d(conv3_5, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_5/dwise')
            print(conv3_5.name, conv3_5.get_shape())
            conv3_5 = slim.convolution2d(conv3_5, 64*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv3_5/linear')
            print(conv3_5.name, conv3_5.get_shape())

            block3_5 = block3_4 + conv3_5
            print(block3_5.name,block3_5.get_shape())

            features['auxiliary_input'] = block3_5

            #28*28*64
            conv4_1 = slim.convolution2d(block3_5, 128, [1, 1], stride=2,scope='conv4_1/expand')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.separable_convolution2d(conv4_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_1/dwise')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.convolution2d(conv4_1, 128*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv4_1/linear')
            print(conv4_1.name, conv4_1.get_shape())

            #14*14*128
            conv5_1 = slim.convolution2d(conv4_1, 512, [1, 1], stride=1,scope='conv5_1/expand')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.separable_convolution2d(conv5_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_1/dwise')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.convolution2d(conv5_1, 128*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv5_1/linear')
            print(conv5_1.name, conv5_1.get_shape())

            conv5_2 = slim.convolution2d(conv5_1, 512, [1, 1], stride=1,scope='conv5_2/expand')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.separable_convolution2d(conv5_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_2/dwise')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.convolution2d(conv5_2, 128*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv5_2/linear')
            print(conv5_2.name, conv5_2.get_shape())

            block5_2 = conv5_1 + conv5_2
            print(block5_2.name,block5_2.get_shape())

            conv5_3 = slim.convolution2d(block5_2, 512, [1, 1], stride=1, scope='conv5_3/expand')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.separable_convolution2d(conv5_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_3/dwise')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.convolution2d(conv5_3, 128*coefficient, [1, 1], stride=1, activation_fn=None, scope='conv5_3/linear')
            print(conv5_3.name, conv5_3.get_shape())

            block5_3 = block5_2 + conv5_3
            print(block5_3.name,block5_3.get_shape())

            conv5_4 = slim.convolution2d(block5_3, 512, [1, 1], stride=1, scope='conv5_4/expand')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.separable_convolution2d(conv5_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_4/dwise')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.convolution2d(conv5_4, 128*coefficient, [1, 1], stride=1, activation_fn=None, scope='conv5_4/linear')
            print(conv5_4.name, conv5_4.get_shape())

            block5_4 = block5_3 + conv5_4
            print(block5_4.name,block5_4.get_shape())

            conv5_5 = slim.convolution2d(block5_4, 512, [1, 1], stride=1, scope='conv5_5/expand')
            print(conv5_5.name, conv5_5.get_shape())
            conv5_5 = slim.separable_convolution2d(conv5_5, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_5/dwise')
            print(conv5_5.name, conv5_5.get_shape())
            conv5_5 = slim.convolution2d(conv5_5, 128*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv5_5/linear')
            print(conv5_5.name, conv5_5.get_shape())

            block5_5 = block5_4 + conv5_5
            print(block5_5.name,block5_5.get_shape())

            conv5_6 = slim.convolution2d(block5_5, 512, [1, 1], stride=1, scope='conv5_6/expand')
            print(conv5_6.name, conv5_6.get_shape())
            conv5_6 = slim.separable_convolution2d(conv5_6, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_6/dwise')
            print(conv5_6.name, conv5_6.get_shape())
            conv5_6 = slim.convolution2d(conv5_6, 128*coefficient, [1, 1], stride=1, activation_fn=None, scope='conv5_6/linear')
            print(conv5_6.name, conv5_6.get_shape())

            block5_6 = block5_5 + conv5_6
            print(block5_6.name,block5_6.get_shape())

            #14*14*128
            conv6_1 = slim.convolution2d(block5_6, 256, [1, 1], stride=1, scope='conv6_1/expand')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.separable_convolution2d(conv6_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_1/dwise')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.convolution2d(conv6_1, 16*coefficient, [1, 1], stride=1, activation_fn=None,scope='conv6_1/linear')
            print(conv6_1.name, conv6_1.get_shape())

            #14*14*16
            conv7 = slim.convolution2d(conv6_1, 32*coefficient, [3, 3], stride=2,scope='conv7')
            print(conv7.name, conv7.get_shape())

            #7*7*32
            conv8 = slim.convolution2d(conv7, 128*coefficient, [7, 7], stride=1,scope='conv8',padding='VALID')
            print(conv8.name, conv8.get_shape())

            avg_pool1 = slim.avg_pool2d(conv6_1, [conv6_1.get_shape()[1], conv6_1.get_shape()[2]], stride=1)
            print(avg_pool1.name, avg_pool1.get_shape())

            avg_pool2 = slim.avg_pool2d(conv7,[conv7.get_shape()[1],conv7.get_shape()[2]],stride=1)
            print(avg_pool2.name,avg_pool2.get_shape())

            s1 = slim.flatten(avg_pool1)
            s2 = slim.flatten(avg_pool2)
            #1*1*128
            s3 = slim.flatten(conv8)
            multi_scale = tf.concat([s1,s2,s3],1)
            landmarks = slim.fully_connected(multi_scale,num_outputs=196,activation_fn=None,scope='fc')
            return features ,landmarks

def create_model(input, landmark, phase_train, args):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections':  None,#tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': phase_train
    }

    landmark_dim = int(landmark.get_shape()[-1])
    features ,landmarks_pre = pfld_inference(input, args.weight_decay, batch_norm_params)
    # loss
    landmarks_loss = tf.reduce_sum(tf.square(landmarks_pre - landmark), axis=1)
    landmarks_loss = tf.reduce_mean(landmarks_loss)

    # add the auxiliary net
    # : finish the loss function
    print('\nauxiliary net')
    with slim.arg_scope([slim.convolution2d, slim.fully_connected], \
                        activation_fn=tf.nn.relu,\
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(args.weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        pfld_input = features['auxiliary_input']
        net_aux = slim.convolution2d(pfld_input, 128, [3, 3], stride=2, scope='pfld_conv1')
        print(net_aux.name,net_aux.get_shape())
        # net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
        net_aux = slim.convolution2d(net_aux, 128,[3,3], stride=1 ,scope='pfld_conv2')
        print(net_aux.name,net_aux.get_shape())
        net_aux = slim.convolution2d(net_aux,32,[3,3],stride=2,scope='pfld_conv3')
        print(net_aux.name,net_aux.get_shape())
        net_aux = slim.convolution2d(net_aux,128,[7,7],stride=1,scope='pfld_conv4')
        print(net_aux.name,net_aux.get_shape())
        net_aux = slim.max_pool2d(net_aux, kernel_size=[3, 3], stride=1, scope='pool1', padding='SAME')
        print(net_aux.name,net_aux.get_shape())
        net_aux = slim.flatten(net_aux)
        print(net_aux.name,net_aux.get_shape())
        fc1 = slim.fully_connected(net_aux,num_outputs=32, activation_fn=None, scope='pfld_fc1')
        print(fc1.name,fc1.get_shape())
        euler_angles_pre = slim.fully_connected(fc1,num_outputs=3, activation_fn=None, scope='pfld_fc2')
        print(euler_angles_pre.name,euler_angles_pre.get_shape())

    # return landmarks_loss, landmarks, heatmap_loss, HeatMaps
    return landmarks_pre,landmarks_loss, euler_angles_pre

