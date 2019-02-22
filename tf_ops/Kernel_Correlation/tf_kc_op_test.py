import tensorflow as tf
import numpy as np
from .kernel_correlation import kernel_correlation, _kernel_correlation_grad


class KernelCorrelationTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:0'):
      tmp1 = np.random.random((8, 32, 64, 3)).astype('float32')
      points = tf.constant(tmp1)
      tmp2 = np.random.random((16, 16, 3)).astype('float32')
      kernel = tf.constant(tmp2)
      sigma = 0.005
      features = kernel_correlation(points, kernel, sigma)

    with self.test_session():
      print("---- Going to compute gradient error")
      err = tf.test.compute_gradient_error(kernel, (16, 16, 3), features, (8, 16, 32))
      print(err)
      self.assertLess(err, 111)


if __name__=='__main__':
  tf.test.main()

