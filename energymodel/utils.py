import tensorflow as tf
from typing import Callable


def random_uniform(size):
  return tf.random.uniform(
      shape=size, minval=-1.0, maxval=1.0, dtype='float32')


def nabla(scalar_fn):
  """Computes ∇f. If the input of f has multiple components, then returns a
  list of ∇f for each component.
  """

  def nabla_fn(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = tf.reduce_sum(scalar_fn(x))
      return tape.gradient(y, x, unconnected_gradients='zero')

  return nabla_fn


def clip_value(fn, threshold):
  threshold = tf.convert_to_tensor(threshold, dtype='float32')

  def clipped_fn(x):
    y = fn(x)
    return tf.nest.map_structure(
        lambda x: tf.clip_by_value(x, -threshold, threshold), y)

  return clipped_fn
