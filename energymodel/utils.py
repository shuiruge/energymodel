import tensorflow as tf
from typing import Callable


def random_uniform(size):
  return tf.random.uniform(
      shape=size, minval=-1.0, maxval=1.0, dtype='float32')


def nabla(scalar_fn: Callable[[tf.Tensor], tf.Tensor]):

  def nabla_fn(x: tf.Tensor):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = tf.reduce_sum(scalar_fn(x))
      return tape.gradient(y, x, unconnected_gradients='zero')

  return nabla_fn


class RandomWalk:
  
  def __init__(self, t, dt, T):
    self._t = tf.Variable(t, dtype='float32', trainable=False)
    self._dt = tf.Variable(dt, dtype='float32', trainable=False)
    self._T = tf.Variable(T, dtype='float32', trainable=False)
  
  @property
  def t(self):
    return self._t
  
  @t.setter
  def t(self, value):
    self._t.assign(value)

  @property
  def dt(self):
    return self._dt
  
  @dt.setter
  def dt(self, value):
    self._dt.assign(value)

  @property
  def T(self):
    return self._T
  
  @T.setter
  def T(self, value):
    self._T.assign(value)
  
  @staticmethod
  def get_proper_T(
      t: float,
      vector_field_order: float,
  ):
    r"""Suppose that f(x) persists its order during evolution, Thus, the proper
    T shall balance the deterministic and the stochastic terms, thus satisfies

        $$ f t \sim \sqrt{2 T t} \implies T \sim f^2 t / 2. $$

    """
    return vector_field_order**2 * t / 2

  @staticmethod
  def get_proper_instance(
      t: float,
      dt: float,
      vector_field_order: float,
  ):
    T = RandomWalk.get_proper_T(t, vector_field_order)
    return RandomWalk(t, dt, T)

  def evolve(
      self,
      f: Callable[[tf.Tensor], tf.Tensor],
      x: tf.Tensor,
  ):
    stddev = tf.sqrt(2 * self.T * self.dt)

    s = tf.constant(0.0)
    while tf.less(s, self.t):
      x += f(x) * self.dt
      if tf.greater(stddev, 0.0):
        x += stddev * tf.random.truncated_normal(tf.shape(x))
      s += self.dt
    return x
  
  def inplace_evolve(
      self,
      f: Callable[[tf.Tensor], tf.Tensor],
      x: tf.Variable,
  ):
    stddev = tf.sqrt(2 * self.T * self.dt)

    s = tf.constant(0.0)
    while tf.less(s, self.t):
      x.assign_add(f(x) * self.dt)
      if tf.greater(stddev, 0.0):
        x.assign_add(stddev * tf.random.truncated_normal(tf.shape(x)))
      s += self.dt
 

def clip_value(
    fn: Callable[[tf.Tensor], tf.Tensor],
    clip_min: float,
    clip_max: float,
):
  clip_min = tf.convert_to_tensor(clip_min, dtype='float32')
  clip_max = tf.convert_to_tensor(clip_max, dtype='float32')

  def clipped_fn(x: tf.Tensor):
    y = fn(x)
    return tf.clip_by_value(y, clip_min, clip_max)
  
  return clipped_fn
