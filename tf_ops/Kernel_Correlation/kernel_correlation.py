import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

KC_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_kc_so.so'))


def kernel_correlation(points, kernel, sigma):
    '''

    :param points: (B, N, npoints, 3)
    :param kernel: (L, m, 3)
    :param sigma: gauss kernel sigma
    npoints: nearest points numbers
    :return: outputs: (B, L, N)
    '''
    return KC_module.kernel_correlation(points, kernel, sigma)
@tf.RegisterGradient('KernelCorrelation')

def _kernel_correlation_grad(op, grad):
    '''

    :param points: (B, N, npoints, 3)
    :param kernel: (L, m, 3)
    :param sigma: gauss kernel sigma


    :return: outputs: (B, L, N)
    '''
    points = op.inputs[0]
    kernel = op.inputs[1]
    # features = op.outputs[0]
    # sigma = op.inputs[2]
    sigma = op.get_attr('sigma')
    grad_kernel = KC_module.kernel_correlation_grad(points, kernel, grad, sigma)
    #grouping_module.group_point_grad(points, idx, grad_out)
    return [None,  grad_kernel]


if __name__=='__main__':
    knn=True
    import numpy as np
    import time
    np.random.seed(100)
    tmp1 = np.random.random((32, 512, 64, 3)).astype('float32')
    tmp2 = np.random.random((16, 16, 3)).astype('float32')
    with tf.device('/gpu:0'):
        points = tf.constant(tmp1)
        kernel = tf.constant(tmp2)
        sigma = 0.005
        features = kernel_correlation(points, kernel, sigma)

    with tf.Session('') as sess:
        now = time.time()
        for _ in range(100):
            ret = sess.run(features)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
        print(ret)


