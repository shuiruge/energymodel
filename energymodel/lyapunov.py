import tensorflow as tf
from typing import Callable

from .models import get_adaptive_temperature, check_nan
from .sde import SDESolver, SDE
from .utils import TensorLike, nest_map


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
      initial: TensorLike,
      solver: SDESolver,
      t: float,
      T: float = None,
  ):
    self.vector_field = vector_field
    self.initial = to_tensor(initial)
    self.solver = solver
    self.t = to_tensor(t)
    if T is None:
      self.T = get_adaptive_temperature(
          vector_field_samples=vector_field(initial),
          t=self.t,
      )
    else:
      self.T = to_tensor(T)

    @nest_map
    def cholesky(s):
      return tf.sqrt(2 * self.T) * s

    self.sde = SDE(
        vector_field=lambda x, t: vector_field(x),
        cholesky=lambda x, t, s: cholesky(s),
    )
    self.particles = initialize_variable(self.initial, trainable=False)

  def __call__(self):
    new_particles = self.solver(self.sde, 0., self.t, self.particles)
    assign_variable(self.particles, new_particles)
    check_nan(self.particles, 'particles')
    return self.particles
