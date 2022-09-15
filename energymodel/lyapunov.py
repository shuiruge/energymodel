import tensorflow as tf
from typing import Callable

from .models import get_adaptive_temperature
from .sde import SDESolver, SDE
from .utils import TensorLike, nest_map, check_nan


@nest_map
def to_tensor(x):
  return tf.convert_to_tensor(x, dtype='float32')


@nest_map
def assign_variable(var, x):
  var.assign(x)


@nest_map
def initialize_variable(x, trainable=None):
  return tf.Variable(x, trainable=trainable)



# TODO: Add docstring.
class Lyapunov:

  def __init__(
      self,
      vector_field: Callable[[TensorLike], TensorLike],
      resample: Callable[[int], TensorLike],
      solver: SDESolver,
      t: float,
      T: float = None,
      batch_size: int = 128,
  ):
    self.vector_field = vector_field
    self.resample = resample
    self.solver = solver
    self.t = to_tensor(t)
    if T is None:
      self.T = get_adaptive_temperature(
          vector_field_samples=vector_field(resample(batch_size)),
          t=self.t,
      )
    else:
      self.T = to_tensor(T)
    self.batch_size = batch_size

    @nest_map
    def cholesky(s):
      return tf.sqrt(2 * self.T) * s

    self.sde = SDE(
        vector_field=lambda x, t: vector_field(x),
        cholesky=lambda x, t, s: cholesky(s),
    )

  def __call__(self):
    particles = self.solver(self.sde, 0., self.t, self.resample(self.batch_size))
    check_nan(particles, 'particles')
    return particles
