from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import LandmarkImage

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

            # 96*96*3
            conv_1 = slim.convolution2d(input, 16, [3, 3], stride=2, scope='conv_1')
            print(conv_1.name, conv_1.get_shape())

            # 48*48*32
            conv2_1 = slim.separable_convolution2d(conv_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv2_1/dwise')
            print(conv2_1.name, conv2_1.get_shape())
            conv2_1 = slim.convolution2d(conv2_1, 8, [1, 1], stride=1, activation_fn=None,
                                         scope='conv2_1/linear')
            print(conv2_1.name, conv2_1.get_shape())
            features['feature2'] = conv2_1
            # 48*48*16
            conv3_1 = slim.convolution2d(conv2_1, 48, [1, 1], stride=1, scope='conv3_1/expand')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.separable_convolution2d(conv3_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_1/dwise')
            print(conv3_1.name, conv3_1.get_shape())
            conv3_1 = slim.convolution2d(conv3_1, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_1/linear')
            print(conv3_1.name, conv3_1.get_shape())

            conv3_2 = slim.convolution2d(conv3_1, 96, [1, 1], stride=1, scope='conv3_2/expand')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.separable_convolution2d(conv3_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv3_2/dwise')
            print(conv3_2.name, conv3_2.get_shape())
            conv3_2 = slim.convolution2d(conv3_2, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv3_2/linear')
            print(conv3_2.name, conv3_2.get_shape())
            block_3_2 = conv3_1 + conv3_2
            print(block_3_2.name, block_3_2.get_shape())

            features['feature3'] = block_3_2
            # 24*24*24
            conv4_1 = slim.convolution2d(block_3_2, 96, [1, 1], stride=1, scope='conv4_1/expand')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.separable_convolution2d(conv4_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_1/dwise')
            print(conv4_1.name, conv4_1.get_shape())
            conv4_1 = slim.convolution2d(conv4_1, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_1/linear')
            print(conv4_1.name, conv4_1.get_shape())

            conv4_2 = slim.convolution2d(conv4_1, 96, [1, 1], stride=1, scope='conv4_2/expand')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.separable_convolution2d(conv4_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_2/dwise')
            print(conv4_2.name, conv4_2.get_shape())
            conv4_2 = slim.convolution2d(conv4_2, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_2/linear')
            print(conv4_2.name, conv4_2.get_shape())
            block_4_2 = conv4_1 + conv4_2
            print(block_4_2.name, block_4_2.get_shape())

            conv4_3 = slim.convolution2d(block_4_2, 96, [1, 1], stride=1, scope='conv4_3/expand')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.separable_convolution2d(conv4_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv4_3/dwise')
            print(conv4_3.name, conv4_3.get_shape())
            conv4_3 = slim.convolution2d(conv4_3, 16, [1, 1], stride=1, activation_fn=None,
                                         scope='conv4_3/linear')
            print(conv4_3.name, conv4_3.get_shape())
            block_4_3 = block_4_2 + conv4_3
            print(block_4_3.name, block_4_3.get_shape())

            # 12*12*32
            features['feature4'] = block_4_3
            conv5_1 = slim.convolution2d(block_4_3, 96, [1, 1], stride=1, scope='conv5_1/expand')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.separable_convolution2d(conv5_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_1/dwise')
            print(conv5_1.name, conv5_1.get_shape())
            conv5_1 = slim.convolution2d(conv5_1, 32, [1, 1], stride=1,activation_fn=None,
                                         scope='conv5_1/linear')
            print(conv5_1.name, conv5_1.get_shape())

            conv5_2 = slim.convolution2d(conv5_1, 192, [1, 1], stride=1, scope='conv5_2/expand')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.separable_convolution2d(conv5_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_2/dwise')
            print(conv5_2.name, conv5_2.get_shape())
            conv5_2 = slim.convolution2d(conv5_2, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_2/linear')
            print(conv5_2.name, conv5_2.get_shape())
            block_5_2 = conv5_1 + conv5_2
            print(block_5_2.name, block_5_2.get_shape())

            conv5_3 = slim.convolution2d(block_5_2, 192, [1, 1], stride=1, scope='conv5_3/expand')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.separable_convolution2d(conv5_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_3/dwise')
            print(conv5_3.name, conv5_3.get_shape())
            conv5_3 = slim.convolution2d(conv5_3, 32, [1, 1], stride=1,  activation_fn=None,
                                         scope='conv5_3/linear')
            print(conv5_3.name, conv5_3.get_shape())
            block_5_3 = block_5_2 + conv5_3
            print(block_5_3.name, block_5_3.get_shape())

            conv5_4 = slim.convolution2d(block_5_3, 192, [1, 1], stride=1, scope='conv5_4/expand')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.separable_convolution2d(conv5_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv5_4/dwise')
            print(conv5_4.name, conv5_4.get_shape())
            conv5_4 = slim.convolution2d(conv5_4, 32, [1, 1], stride=1, activation_fn=None,
                                         scope='conv5_4/linear')
            print(conv5_4.name, conv5_4.get_shape())
            block_5_4 = block_5_3 + conv5_4
            print(block_5_4.name, block_5_4.get_shape())

            # 6*6*64
            conv6_1 = slim.convolution2d(block_5_4, 192, [1, 1], stride=1, scope='conv6_1/expand')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.separable_convolution2d(conv6_1, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_1/dwise')
            print(conv6_1.name, conv6_1.get_shape())
            conv6_1 = slim.convolution2d(conv6_1, 48, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_1/linear')
            print(conv6_1.name, conv6_1.get_shape())

            conv6_2 = slim.convolution2d(conv6_1, 288, [1, 1], stride=1, scope='conv6_2/expand')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.separable_convolution2d(conv6_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_2/dwise')
            print(conv6_2.name, conv6_2.get_shape())
            conv6_2 = slim.convolution2d(conv6_2, 48, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_2/linear')
            print(conv6_2.name, conv6_2.get_shape())
            block_6_2 = conv6_1 + conv6_2
            print(block_6_2.name, block_6_2.get_shape())

            conv6_3 = slim.convolution2d(block_6_2, 288, [1, 1], stride=1, scope='conv6_3/expand')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.separable_convolution2d(conv6_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv6_3/dwise')
            print(conv6_3.name, conv6_3.get_shape())
            conv6_3 = slim.convolution2d(conv6_3, 48, [1, 1], stride=1, activation_fn=None,
                                         scope='conv6_3/linear')
            print(conv6_3.name, conv6_3.get_shape())
            block_6_3 = block_6_2 + conv6_3
            print(block_6_3.name, block_6_3.get_shape())

            features['feature5'] = block_6_3
            # 6*6*96
            conv7_1 = slim.convolution2d(block_6_3, 288, [1, 1], stride=1, scope='conv7_1/expand')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.separable_convolution2d(conv7_1, num_outputs=None, stride=2, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_1/dwise')
            print(conv7_1.name, conv7_1.get_shape())
            conv7_1 = slim.convolution2d(conv7_1, 80, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_1/linear')
            print(conv7_1.name, conv7_1.get_shape())

            conv7_2 = slim.convolution2d(conv7_1, 480, [1, 1], stride=1, scope='conv7_2/expand')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.separable_convolution2d(conv7_2, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_2/dwise')
            print(conv7_2.name, conv7_2.get_shape())
            conv7_2 = slim.convolution2d(conv7_2, 80, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_2/linear')
            print(conv7_2.name, conv7_2.get_shape())
            block_7_2 = conv7_1 + conv7_2
            print(block_7_2.name, block_7_2.get_shape())


            conv7_3 = slim.convolution2d(block_7_2, 480, [1, 1], stride=1, scope='conv7_3/expand')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.separable_convolution2d(conv7_3, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_3/dwise')
            print(conv7_3.name, conv7_3.get_shape())
            conv7_3 = slim.convolution2d(conv7_3, 80, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_3/linear')
            print(conv7_3.name, conv7_3.get_shape())
            block_7_3 = block_7_2 + conv7_3
            print(block_7_3.name, block_7_3.get_shape())

            conv7_4 = slim.convolution2d(block_7_3, 480, [1, 1], stride=1, scope='conv7_4/expand')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.separable_convolution2d(conv7_4, num_outputs=None, stride=1, depth_multiplier=1,
                                                   kernel_size=[3, 3], scope='conv7_4/dwise')
            print(conv7_4.name, conv7_4.get_shape())
            conv7_4 = slim.convolution2d(conv7_4, 256, [1, 1], stride=1, activation_fn=None,
                                         scope='conv7_4/linear')
            print(conv7_4.name, conv7_4.get_shape())
            features['feature6'] = conv7_4
    return features

def create_model(input, landmark, phase_train, args):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections':  None,#tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': phase_train
    }

    landmark_dim = int(landmark.get_shape()[-1])
    features = mobilenet_v2(input, args.weight_decay, batch_norm_params)

    L1_in = feature6 = features['feature6']
    L2_in = feature5 = features['feature5']
    L3_in = feature4 = features['feature4']
    L4_in = feature3 = features['feature3']
    L5_in = feature2 = features['feature2']

    print('\n##### featrue\n')
    print('feature6({}): {}'.format(feature6.name, feature6.get_shape()))
    print('feature5({}): {}'.format(feature5.name, feature5.get_shape()))
    print('feature4({}): {}'.format(feature4.name, feature4.get_shape()))
    print('feature3({}): {}'.format(feature3.name, feature3.get_shape()))
    print('feature2({}): {}'.format(feature2.name, feature2.get_shape()))
    print('\n-----------\n')
    # L1_in = tf.stop_gradient(tf.identity(L1_in, 'L1_in_stop_gradient'))
    # L2_in = tf.stop_gradient(tf.identity(L2_in, 'L2_in_stop_gradient'))
    # L3_in = tf.stop_gradient(tf.identity(L3_in, 'L3_in_stop_gradient'))
    # L4_in = tf.stop_gradient(tf.identity(L4_in, 'L4_in_stop_gradient'))
    # L5_in = tf.stop_gradient(tf.identity(L5_in, 'L5_in_stop_gradient'))

    print('L1_in({}): {}'.format(L1_in.name, L1_in.get_shape()))
    print('L2_in({}): {}'.format(L2_in.name, L2_in.get_shape()))
    print('L3_in({}): {}'.format(L3_in.name, L3_in.get_shape()))
    print('L4_in({}): {}'.format(L4_in.name, L4_in.get_shape()))
    print('L5_in({}): {}'.format(L5_in.name, L5_in.get_shape()))
    print('\n##### featrue\n')

    with slim.arg_scope([slim.convolution2d, slim.fully_connected], \
                        activation_fn=tf.nn.relu,\
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(args.weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        #Level 1
        print('Level 1')
        net = slim.avg_pool2d(L1_in, L1_in.get_shape()[1:3], padding='VALID', scope='L1_avg_pool')
        F1 = L1_in
        print(net.name, net.get_shape())
        net = slim.flatten(net)
        landmark_L1 = slim.fully_connected(net, num_outputs=landmark_dim, activation_fn=None, scope='L1_fc')
        print(landmark_L1.name, landmark_L1.get_shape())

        #Level 2
        print('\nLevel 2')
        image_size = L2_in.get_shape()
        sigma = tf.to_float(tf.reduce_max(image_size[1:3])) / 2
        L2_HeatMap = LandmarkImage(landmark_L1, image_size, sigma=sigma)
        print(L2_HeatMap.name, L2_HeatMap.get_shape())
        L2_HeatMap = tf.tile(tf.expand_dims(L2_HeatMap,-1),(1, 1, 1, image_size[3]))
        print(L2_HeatMap.name, L2_HeatMap.get_shape())
        L2_HeatMap = tf.identity(tf.stop_gradient(L2_HeatMap), 'L2_HeatMap')
        print(L2_HeatMap.name, L2_HeatMap.get_shape())

        height = L2_in.get_shape().as_list()[1]
        width = L2_in.get_shape().as_list()[2]
        chn = L2_in.get_shape().as_list()[3]

        concat2 = slim.convolution2d(F1, chn, [1, 1], stride=1)
        print(concat2.name, concat2.get_shape())
        concat2 = tf.image.resize_images(concat2, [height, width], method=0)
        print(concat2.name, concat2.get_shape())
        #添加L1 in
        net = L2_in * L2_HeatMap
        net = tf.concat([net, concat2], 3)
        print(net.name, net.get_shape())

        net = slim.convolution2d(net, 96, [1, 1], stride=1, scope='L2_conv1')
        print(net.name, net.get_shape())
        net = slim.separable_convolution2d(net, num_outputs=None, stride=1, depth_multiplier=1,
                                               kernel_size=[3, 3], scope='L2_conv/dwise')
        print(net.name, net.get_shape())
        net = slim.convolution2d(net, 256, [1, 1], stride=1, scope='L2_conv2')
        print(net.name, net.get_shape())
        F2 = net
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='L2_avg_pool')
        print(net.name, net.get_shape())
        net = slim.flatten(net)
        print(net.name, net.get_shape())

        landmark_L2 = slim.fully_connected(net, num_outputs=landmark_dim, activation_fn=None, scope='L2_fc')
        print(landmark_L2.name, landmark_L2.get_shape())

        # Level 3
        print('\nLevel 3')
        image_size = L3_in.get_shape()
        sigma = tf.to_float(tf.reduce_max(image_size[1:3])) / 3
        # L3_HeatMap = LandmarkImage(landmark_L2, image_size, sigma=sigma)
        L3_HeatMap = LandmarkImage(landmark_L1, image_size, sigma=sigma)
        print(L3_HeatMap.name, L3_HeatMap.get_shape())
        L3_HeatMap = tf.tile(tf.expand_dims(L3_HeatMap, -1), (1, 1, 1, image_size[3]))
        print(L3_HeatMap.name, L3_HeatMap.get_shape())
        L3_HeatMap = tf.identity(tf.stop_gradient(L3_HeatMap), 'L3_HeatMap')
        print(L3_HeatMap.name, L3_HeatMap.get_shape())

        height = L3_in.get_shape().as_list()[1]
        width = L3_in.get_shape().as_list()[2]
        chn = L3_in.get_shape().as_list()[3]
        concat3 = slim.convolution2d(F2, chn, [1, 1], stride=1)
        print(concat3.name, concat3.get_shape())
        concat3 = tf.image.resize_images(concat3, [height, width], method=0)
        print(concat3.name, concat3.get_shape())

        net = L3_in * L3_HeatMap
        net = tf.concat([net, concat3], 3)
        print(net.name, net.get_shape())

        net = slim.convolution2d(net, 32, [1, 1], stride=1, scope='L3_conv1')
        print(net.name, net.get_shape())
        net = slim.separable_convolution2d(net, num_outputs=None, stride=1, depth_multiplier=1,
                                           kernel_size=[3, 3], scope='L3_conv/dwise')
        print(net.name, net.get_shape())
        net = slim.convolution2d(net, 256, [1, 1], stride=1, scope='L3_conv2')
        F3 = net
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='L3_avg_pool')
        print(net.name, net.get_shape())
        net = slim.flatten(net)
        print(net.name, net.get_shape())
        landmark_L3 = slim.fully_connected(net, num_outputs=landmark_dim, activation_fn=None, scope='L3_fc')
        print(landmark_L3.name, landmark_L3.get_shape())


        # Level 4
        print('\nLevel 4')
        image_size = L4_in.get_shape()
        sigma = tf.to_float(tf.reduce_max(image_size[1:3])) / 4
        # L4_HeatMap = LandmarkImage(landmark_L3, image_size, sigma=sigma)
        L4_HeatMap = LandmarkImage(landmark_L1, image_size, sigma=sigma)
        print(L4_HeatMap.name, L4_HeatMap.get_shape())
        L4_HeatMap = tf.tile(tf.expand_dims(L4_HeatMap, -1), (1, 1, 1, image_size[3]))
        print(L4_HeatMap.name, L4_HeatMap.get_shape())
        L4_HeatMap = tf.identity(tf.stop_gradient(L4_HeatMap), 'L4_HeatMap')
        print(L4_HeatMap.name, L4_HeatMap.get_shape())

        height = L4_in.get_shape().as_list()[1]
        width = L4_in.get_shape().as_list()[2]
        chn = L4_in.get_shape().as_list()[3]
        concat4 = slim.convolution2d(F3, chn, [1, 1], stride=1)
        print(concat4.name, concat4.get_shape())
        concat4 = tf.image.resize_images(concat4, [height, width], method=0)
        print(concat4.name, concat4.get_shape())

        net = L4_in * L4_HeatMap
        net = tf.concat([net, concat4], 3)
        print(net.name, net.get_shape())

        net = slim.convolution2d(net, 24, [1, 1], stride=1, scope='L4_conv1')
        print(net.name, net.get_shape())
        net = slim.separable_convolution2d(net, num_outputs=None, stride=1, depth_multiplier=1,
                                           kernel_size=[3, 3], scope='L4_conv/dwise')
        print(net.name, net.get_shape())
        net = slim.convolution2d(net, 256, [1, 1], stride=1, scope='L4_conv2')
        F4 = net
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='L4_avg_pool')
        print(net.name, net.get_shape())
        net = slim.flatten(net)
        print(net.name, net.get_shape())
        landmark_L4 = slim.fully_connected(net, num_outputs=landmark_dim, activation_fn=None, scope='L4_fc')
        print(landmark_L4.name, landmark_L4.get_shape())

        # Level 5
        print('\nLevel 5')
        image_size = L5_in.get_shape()
        sigma = tf.to_float(tf.reduce_max(image_size[1:3])) / 5
        # L5_HeatMap = LandmarkImage(landmark_L4, image_size, sigma=sigma)
        L5_HeatMap = LandmarkImage(landmark_L1, image_size, sigma=sigma)
        print(L5_HeatMap.name, L5_HeatMap.get_shape())
        L5_HeatMap = tf.tile(tf.expand_dims(L5_HeatMap, -1), (1, 1, 1, image_size[3]))
        print(L5_HeatMap.name, L5_HeatMap.get_shape())
        L5_HeatMap = tf.identity(tf.stop_gradient(L5_HeatMap), 'L5_HeatMap')
        print(L5_HeatMap.name, L5_HeatMap.get_shape())

        height = L5_in.get_shape().as_list()[1]
        width = L5_in.get_shape().as_list()[2]
        chn = L5_in.get_shape().as_list()[3]
        concat5 = slim.convolution2d(F4, chn, [1, 1], stride=1)
        print(concat5.name, concat5.get_shape())
        concat5 = tf.image.resize_images(concat5, [height, width], method=0)
        print(concat5.name, concat5.get_shape())

        net = L5_in * L5_HeatMap
        net = tf.concat([net, concat5], 3)
        print(net.name, net.get_shape())

        net = slim.convolution2d(net, 16, [1, 1], stride=1, scope='L5_conv1')
        print(net.name, net.get_shape())
        net = slim.separable_convolution2d(net, num_outputs=None, stride=1, depth_multiplier=1,
                                           kernel_size=[3, 3], scope='L5_conv/dwise')
        print(net.name, net.get_shape())
        net = slim.convolution2d(net, 256, [1, 1], stride=1, scope='L5_conv2')
        F5 = net
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='L5_avg_pool')
        print(net.name, net.get_shape())
        net = slim.flatten(net)
        print(net.name, net.get_shape())
        landmark_L5 = slim.fully_connected(net, num_outputs=landmark_dim, activation_fn=None, scope='L5_fc')
        print(landmark_L5.name, landmark_L5.get_shape())

        # Label
        print('\nLabel')
        image_size = L5_in.get_shape()
        sigma = tf.to_float(tf.reduce_max(image_size[1:3])) / 5
        Label_HeatMap = LandmarkImage(landmark, image_size, sigma=sigma)
        print(Label_HeatMap.name, Label_HeatMap.get_shape())
        Label_HeatMap = tf.expand_dims(Label_HeatMap, -1)
        print(Label_HeatMap.name, Label_HeatMap.get_shape())
        Label_HeatMap = tf.identity(tf.stop_gradient(Label_HeatMap), 'Label_HeatMap')
        print(Label_HeatMap.name, Label_HeatMap.get_shape())

        heatmap = slim.convolution2d(F5, 1, [3, 3], stride=1, activation_fn=None, scope='heatmap_out')
        print(heatmap.name, heatmap.get_shape())

        # loss
        L1_loss = tf.reduce_sum(tf.square(landmark_L1 - landmark), axis=1)
        L2_loss = tf.reduce_sum(tf.square(landmark_L2 - landmark), axis=1)
        L3_loss = tf.reduce_sum(tf.square(landmark_L3 - landmark), axis=1)
        L4_loss = tf.reduce_sum(tf.square(landmark_L4 - landmark), axis=1)
        L5_loss = tf.reduce_sum(tf.square(landmark_L5 - landmark), axis=1)

        landmarks_loss = [tf.reduce_mean(L1_loss), \
                          tf.reduce_mean(L2_loss), \
                          tf.reduce_mean(L3_loss), \
                          tf.reduce_mean(L4_loss), \
                          tf.reduce_mean(L5_loss)]

        heatmap_loss = tf.reduce_sum(tf.square(heatmap - Label_HeatMap), axis=(1,2,3))
        heatmap_loss = tf.reduce_mean(heatmap_loss)

    landmark_L1 = tf.identity(landmark_L1, 'landmark_L1')
    landmark_L2 = tf.identity(landmark_L2, 'landmark_L2')
    landmark_L3 = tf.identity(landmark_L3, 'landmark_L3')
    landmark_L4 = tf.identity(landmark_L4, 'landmark_L4')
    landmark_L5 = tf.identity(landmark_L5, 'landmark_L5')
    landmarks = [landmark_L1, landmark_L2, landmark_L3, landmark_L4, landmark_L5]
    HeatMaps = [L2_HeatMap, L3_HeatMap, L4_HeatMap, L5_HeatMap, heatmap, Label_HeatMap]

    return landmarks_loss, landmarks, heatmap_loss, HeatMaps

