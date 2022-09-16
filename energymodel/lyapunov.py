import tensorflow as tf
from typing import Callable

from .models import get_adaptive_temperature
from .sde import SDESolver, SDE
from .utils import ScalarLike, TensorLike, nest_map, check_nan


# TODO: Add docstring.
class Lyapunov:

  def __init__(
      self,
      vector_field: Callable[[TensorLike], TensorLike],
      resample: Callable[[int], TensorLike],
      solver: SDESolver,
      t: ScalarLike,
      T: ScalarLike,
  ):
    self.vector_field = vector_field
    self.resample = resample
    self.solver = solver
    self.t = tf.convert_to_tensor(t, dtype='float32')
    self.T = tf.convert_to_tensor(T, dtype='float32')

    @nest_map
    def cholesky(s):
      return tf.sqrt(2 * self.T) * s

    self.sde = SDE(
        vector_field=lambda x, t: vector_field(x),
        cholesky=lambda x, t, s: cholesky(s),
    )

  def __call__(self, batch_size: int):
    particles = self.solver(
        sde=self.sde,
        t0=0.,
        t1=self.t,
        x0=self.resample(batch_size),
    )
    check_nan(particles, 'particles')
    return particles
