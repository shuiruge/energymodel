import tensorflow as tf
from .utils import nest_map


@nest_map
def random_seed(x, stddev):
  return tf.random.truncated_normal(tf.shape(x), 0., stddev)


class SDE:
  r"""Stochastic differential equations (SDE):

  \begin{equation}
    dx^a = f^a(x, t) dt + dW^a(x, t),
  \end{equation}

  where $dW^a \sim \text{Normal}(0, K^{ab}(x,t) dt)$.

  By Cholesky decomposition, we can sample $dW$ by first sampling seed from
  $S_a = \text{Normal}(0, dt)$, and then $dW^a = L^{ab}(x,t) S_b$, where
  $K = L^T L$.
  """

  def __init__(self, vector_field, cholesky):
    """
    Args:
      vector_field: Defines the $f(x, t)$, where x is tensor or nested tensor,
        and returns the same type as x.
      cholesky: Defines the $L^{ab}(x, t) S_b$ where S is random seed sampled
        from $\text{Normal}(0, dt)$. It has signature `cholesky(x, t, seed)`.
        The x, the seed, and the return value are tensors or nested tensors,
        the same type.
    """
    self.vector_field = vector_field
    self.cholesky = cholesky

  def evolve(self, t0, t1, dt, x0):
    """Evolves teh SDE by Euler method.

    Args:
        t0: The starting time. Scalar.
        t1: The end time. Scalar.
        dt: The time step. Scalar.
        x0: The initial value x(t0). Tensor or nested tensor.

    Returns:
        The evolved value x(t1). The same type as `x0`.
    """

    @nest_map
    def euler_step(x, f, dt, dW):
      return x + f*dt + dW

    x = x0
    s = t0
    while tf.less(s, t1):
      f = self.vector_field(x, s)

      seed = random_seed(x, tf.sqrt(dt))
      dW = self.cholesky(x, s, seed)

      x = euler_step(x, f, dt, dW)
      s += dt
    return x
