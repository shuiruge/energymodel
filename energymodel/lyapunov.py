from typing import Callable

import tensorflow as tf

from .sde import SDE, SDESolver
from .utils import ScalarLike, TensorLike, check_nan, nest_map


class Lyapunov:
  """A data generator that generates particles obeying the equilibrium
  distribution of the stochastic differential equation (SDE):

    dx = f(x) dt + dW, dW ~ Normal(0, 2T*dt).

  This is for generating the training data for an energy model that fits a
  Lyapunov function of the vector field `f`.

  Methods:
    __call__: Generates particles.
  """

  def __init__(
      self,
      vector_field: Callable[[TensorLike], TensorLike],
      resample: Callable[[int], TensorLike],
      solver: SDESolver,
      t: ScalarLike,
      T: ScalarLike,
  ):
    """
    Args:
      vector_field: The `f` function.
      resample: The fantasy particles are resampled before sampling by
        evolving SDE. Signature `(batch_size: int) -> particles`, where the
        `particles` is tensor or nested tensor.
      solver: SDE solver.
      t: Time interval of SDE evolution.
      T: The "temperature" `T`.
    """
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
    """Generates particles."""
    particles = self.solver(
        sde=self.sde,
        t0=0.,
        t1=self.t,
        x0=self.resample(batch_size),
    )
    check_nan(particles, 'particles')
    return particles
