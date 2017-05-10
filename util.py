import tensorflow as tf
import numpy as np
import os
from scipy import misc, io


def batch_norm(x, shape, phase_train, scope='BN'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the spatial dimensions of the input tensor.

    Args:
        x:           Tensor,  B...C input maps (e.g. BHWC or BXYZC)
        shape:       Tuple, shape of input
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope

    Returns:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        n_out = shape[-1]  # depth of input maps
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='Beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='Gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(shape[:-1]))), name='Moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv(x, input_shape, num_features, phase_train, do_bn=True, size=3, seed=None, scope='Conv'):
    with tf.variable_scope(scope):
        kernel_shape = [size]*(len(input_shape)-2)
        kernel_shape.append(input_shape[-1])
        kernel_shape.append(num_features)
        # example: input_shape is BHWC, kernel_shape is [3,3,D,num_features]
        kernel = tf.Variable(tf.random_normal(kernel_shape, seed=seed, name='Kernel'))
        convolved = tf.nn.convolution(x, kernel, padding="SAME", name='Conv')
        convolved_shape = list(input_shape)
        convolved_shape[-1] = num_features
        # example: input_shape is BHWC, convolved_shape is [B,H,W,num_features]
        if do_bn:
            return batch_norm(convolved, convolved_shape, phase_train), convolved_shape
        else:
            return convolved, convolved_shape


def relu(x, scope='Relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(x, name='Relu')


def pool(x, input_shape, scope='Pool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = tf.nn.pool(x, window_shape=window_shape, pooling_type="MAX", strides=window_shape, padding="SAME")
        output_shape = [input_shape[0]] + [i / 2 for i in input_shape[1:-1]] + [input_shape[-1]]
        mask = nearest_neighbor(output)
        mask = tf.equal(x, mask)
        mask = tf.cast(mask, tf.float32)
        return output, output_shape, mask


def unpool(x, input_shape, mask, scope='Unpool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = nearest_neighbor(x) * mask
        output_shape = [input_shape[0]] + [i*2 for i in input_shape[1:-1]] + [input_shape[-1]]
        return output, output_shape


def nearest_neighbor_2d(x):
    s = x.get_shape().as_list()
    n = s[1]
    c = s[-1]
    y = tf.tile(x, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n * n, 1, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n, 2 * n, c])
    return y


def nearest_neighbor_3d(x):
    s = x.get_shape().as_list()
    n = s[1]
    c = s[-1]
    y = tf.transpose(x, [0, 3, 1, 2, 4])
    y = tf.reshape(y, [-1, n, n * n, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n * n, n, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 4 * n * n * n, 1, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n, 2 * n, 2 * n, c])
    y = tf.transpose(y, [0, 2, 3, 1, 4])
    return y


def gen_occupancy_grid(x, lower_left, upper_right, divisions):
    output = np.zeros(divisions)
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = x - lower_left
    indices = np.floor(offsets / intervals)
    indices = indices.astype(int)
    print(indices)
    for row in indices:
        print(row)
        if np.sum(row >= np.zeros([1, 3])) == 3 and np.sum(row < divisions) == 3:
            output[row[0], row[1], row[2]] = 1
    return output

class DataReader(object):
    def __init__(self, path, image_shape):
        self._image_shape = image_shape
        self._path = path
        self._image_data = self.get_filenames(path + '/image_data/training/')
        self._image_labels = self.get_filenames(path + '/image_labels/training/')
        self._velodyne_data = self.get_filenames(path + '/velodyne_data/training/')
        self._velodyne_labels = self.get_filenames(path + '/velodyne_labels/training/')

    def get_filenames(self, path):
        data_paths = os.listdir(path)
        data_paths = sorted(data_paths)
        data_paths = [path + data_path for data_path in data_paths]
        filenames = []
        for data_path in data_paths:
            _filenames = os.listdir(data_path)
            _filenames = sorted(_filenames)
            _filenames = [data_path + '/' + filename for filename in _filenames]
            filenames += _filenames
        return filenames

    def get_image_data(self):
        shape = (len(self._image_data),) + self._image_shape
        image_data = np.zeros(shape)
        k = 0
        h, w, c = self._image_shape
        for filename in self._image_data:
            image = misc.imread(filename)
            image_data[k,:,:,:] = image[0:h,0:w,0:c]
            k += 1
        return image_data

    def get_image_labels(self):
        h, w, _ = self._image_shape
        shape = (len(self._image_labels), h, w)
        label_data = np.zeros(shape)
        k = 0
        for filename in self._image_labels:
            label = io.loadmat(filename)
            label = label['truth']
            label_data[k,:,:] = label[0:h,0:w]
            k += 1
        return label_data

dr = DataReader('/home/vdd6/Desktop/gen_seg_data', (374, 1238, 3))
res = dr.get_image_data()
res = dr.get_image_labels()
