import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def train_model(loss, global_step, data_num, args):
    lr_factor = 0.1
    lr_epoch = args.lr_epoch.strip().split(',')
    lr_epoch = list(map(int, lr_epoch))
    boundaries = [epoch*data_num//args.batch_size for epoch in lr_epoch]
    lr_values = [args.learning_rate*(lr_factor**x) for x in range(0, len(lr_epoch)+1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.AdamOptimizer(lr_op)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    return train_op, lr_op

def GaussianMaps(sigma):
    d = int(3 * sigma + 0.5)
    cx, cy = (d, d)
    Pixels = np.zeros((2 * d, 2 * d, 2), dtype=np.int32)
    value = np.zeros((2 * d, 2 * d, 1), dtype=np.float32)
    for x in range(2 * d):
        for y in range(2 * d):
            D = (x - cx) ** 2 + (y - cy) ** 2
            Pixels[y,x] = (x - cx, y - cy)
            if D < (3 * sigma) ** 2:
                value[y,x] = np.exp(-D / 2.0 / sigma / sigma)
            else:
                value[y,x] = 0.0
    return Pixels, value, (2*d,2*d)

def LandmarkImage(Landmarks, image_size, sigma=None):
    if sigma is None:
        sigma = tf.to_float(tf.reduce_max(image_size[1:3]))/4
    d = tf.to_int32(3 * sigma + 0.5)
    xx = tf.tile(tf.expand_dims(tf.range(-d, d, 1), 0), (2 * d, 1))
    yy = tf.tile(tf.expand_dims(tf.range(-d, d, 1), 1), (1, 2 * d))
    Pixels = tf.concat([tf.expand_dims(yy,-1), tf.expand_dims(xx,-1)], axis=-1)
    D = tf.reduce_sum(tf.square(tf.to_float(Pixels)), axis=-1)
    zeros = tf.zeros((2 * d, 2 * d), dtype=tf.float32)
    Gaussian = tf.exp(-D/(2*sigma*sigma))
    values = tf.where(tf.greater(D, (3 * sigma) ** 2), zeros, Gaussian)

    shape = tf.to_float(tf.expand_dims(image_size[1:3],axis=0))

    def Do(L):
        def DoIn(Point):
            intPoint = tf.to_int32(Point)
            locations = Pixels + intPoint
            img = tf.scatter_nd(locations, values, shape=(image_size[1], image_size[2]))
            return img

        L = tf.reverse(tf.reshape(L, [-1, 2]), [-1])*shape
        L = tf.map_fn(DoIn, L)
        L = tf.reshape(tf.reduce_max(L, axis=0), (image_size[1], image_size[2]))
        return L
    Landmarks = tf.clip_by_value(Landmarks, 0, 1)
    return tf.map_fn(Do, Landmarks)    
 
def LandmarkImage_98(Landmarks, image_size, sigma=None):
    if sigma is None:
        sigma = tf.to_float(tf.reduce_max(image_size[1:3]))/4
    d = tf.to_int32(3 * sigma + 0.5)
    xx = tf.tile(tf.expand_dims(tf.range(-d, d, 1), 0), (2 * d, 1))
    yy = tf.tile(tf.expand_dims(tf.range(-d, d, 1), 1), (1, 2 * d))
    Pixels = tf.concat([tf.expand_dims(yy,-1), tf.expand_dims(xx,-1)], axis=-1)
    D = tf.reduce_sum(tf.square(tf.to_float(Pixels)), axis=-1)
    zeros = tf.zeros((2 * d, 2 * d), dtype=tf.float32)
    Gaussian = tf.exp(-D/(2*sigma*sigma))
    values = tf.where(tf.greater(D, (3 * sigma) ** 2), zeros, Gaussian)

    shape = tf.to_float(tf.expand_dims(image_size[1:3],axis=0))
    #print('debug 0 ::','shape value :',shape) #(1,2)

    def Do(L):
        def DoIn(Point):
            intPoint = tf.to_int32(Point)
            locations = Pixels + intPoint
            #print('debug 2','intpoint shape:{} Pixels shape{} locations shape{}'.format(intPoint.shape,Pixels.shape,locations.shape))
            #(2,) (,,2) (,,2) 
            img = tf.scatter_nd(locations, values, shape=(image_size[1], image_size[2]))
            #print('debug 3','img shape:',img.shape)
            #(56,56)
            return img

        L = tf.reverse(tf.reshape(L, [-1, 2]), [-1])*shape
        #print('debug 5',L.shape)
        #(98,2)
        L = tf.map_fn(DoIn, L)
        # L = tf.reshape(tf.reduce_max(L, axis=0), (image_size[1], image_size[2]))
        L = tf.transpose(L,[1,2,0])
        #print('debug 4:','L shape:',L.shape)
        #(56,56)
        return L
    Landmarks = tf.clip_by_value(Landmarks, 0, 1)
    #print('debug 1 ::','Landmarks shape',Landmarks.shape) #(n,196)
    return tf.map_fn(Do, Landmarks)
 
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    landmarks_placeholder = tf.placeholder(tf.float32, shape=(None, 18), name='landmarks')
    landmarks = np.asarray([[0, 0, 0, 5, 0, 10,\
                             5, 0, 5, 5, 5, 10,\
                             10,0, 10,5, 10,10]], dtype=np.float32)/16

    img = LandmarkImage(landmarks_placeholder, (1,16,16,1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(img,feed_dict={landmarks_placeholder:landmarks})
        # print(out)
        print(out.shape)
        # for img in out:
            # plt.xticks([])
            # plt.yticks([])
            # plt.imshow(img,cmap=plt.cm.hot)
            # plt.show()
            # plt.clf()
        image=out[0]*255
        for i in range(image.shape[2]):
            print(image[:,:,i].shape)
            # cv2.imwrite('./result{}.png'.format(i),image[:,:,i])
        