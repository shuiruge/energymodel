import abc
import tensorflow as tf
from typing import Callable
from .utils import TensorLike, map_structure, nest_map, maximum


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

  def __init__(
      self,
      vector_field: Callable[[TensorLike, float], TensorLike],
      cholesky: Callable[[TensorLike, float, TensorLike], TensorLike],
  ):
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


class SDESolver(abc.ABC):

  @abc.abstractmethod
  def __call__(self,
               sde: SDE,
               t0: float,
               t1: float,
               x0: TensorLike) -> TensorLike:
    """Evolves teh SDE.

    Args:
      t0: The starting time.
      t1: The end time.
      x0: The initial value x(t0). Tensor or nested tensor.

    Returns:
      The evolved value x(t1). The same type as `x0`.
    """
    pass


class EMSolver(SDESolver):
  """Euler-Maruyama method for SDE.

  When employ adaptive time step size, we use the theorem:

    If f(x,t) is constant, the evolution of x(t) is independent of the time
    step size.

  In the case f(x,t) = 0 constantly, this is the "Brownian rescaling", a famous
  property of Wiener processes. And we suppose that this is true when f(x,t) is
  approximately constant.

  Given a general f(x,t), for every x, there exists a neighborhood of x such
  that f(x,t) is approximately constant. The maximal range of this neighborhood
  is thus regarded as the adapted time step size.

  Let eps the tolerance of violating constant f(x,t). We are to estimate the
  maximal dt such that |f(x+dx,t+dt) - f(x,t)| <= eps. Here we assume that this
  this difference is linear dependent on dt. With this assumption, the maximal
  dt can be estimated as follows. First given the previous position x-dx and
  the current position x, we can estimate the difference

    |f(x+dx,t+dt) - f(x,t)| ~ |f(x,t) - f(x-dx,t-dt)|.

  Then, use it for estimating the next time step size dt', based on the
  previously assumed linear dependence, as

    |f(x,t) - f(x-dx,t-dt)| dt' / dt = eps.

  That is, dt' = dt |f(x,t) - f(x-dx,t-dt)| / eps.
  """

  def __init__(self,
               dt: float,
               eps: float = None,
               max_dt: float = None):
    """
    Args:
      dt: The time step size. If use adaptive step size, this will be the
        initial value (the first step size) as well as the minimum.
      eps: The tolerance of violating constant f(x,t). If `None`, use constant
        time step size.
      max_dt: The maximum time step size if adaptive. If `None`, use `10*dt`.
    """
    self.dt = tf.convert_to_tensor(dt, dtype='float32')
    if eps is None:
      self.eps = None
    else:
      self.eps = tf.convert_to_tensor(eps, dtype='float32')
    if max_dt is None:
      self.max_dt = 10 * self.dt
    else:
      self.max_dt = tf.convert_to_tensor(max_dt, dtype='float32')

  def __call__(self, sde, t0, t1, x0):

    @nest_map
    def euler_step(x, f, dt, dW):
      return x + f*dt + dW

    x = x0
    s = t0
    dt = self.dt
    f = sde.vector_field(x, s)
    previous_f = f
    while tf.less(s, t1):
      # Determine the time step size `dt`
      if tf.greater(s, t0):
        f = sde.vector_field(x, s)
        if self.eps is not None:
          df = diff(previous_f, f)
          dt *= self.eps / (df + 1e-8)
          dt = tf.clip_by_value(dt, self.dt, self.max_dt)
      if tf.greater(s + dt, t1):
        # Shall be end at `t1` exactly.
        dt = t1 - s

      seed = random_seed(x, tf.sqrt(dt))
      dW = sde.cholesky(x, s, seed)

      x = euler_step(x, f, dt, dW)
      previous_f = f
      s += dt
    return x


@nest_map
def random_seed(x: TensorLike, stddev: float) -> TensorLike:
  return tf.random.truncated_normal(tf.shape(x), 0., stddev)


def infinity_norm(x: tf.Tensor) -> float:
  return tf.reduce_mean(tf.abs(x))


def diff(x: TensorLike, y: TensorLike) -> float:
  _diff = map_structure(lambda x, y: infinity_norm(x - y), x, y)
  return maximum(*tf.nest.flatten(_diff))

