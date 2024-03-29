import abc
from typing import Callable, List

import tensorflow as tf
from tensorflow.python import keras

from .sde import SDE, SDESolver
from .utils import ScalarLike, TensorLike, map_structure, minimum, nest_map


class Callback(abc.ABC):
  """Defines the abstract callback API.

  A callback shall implement the `__call__` method, which has arguments:
    step: An int64 scalar for training step.
    real_particles: Tensor for input batch.
    fantasy_particles: Tensor for input batch.
    loss: Scalar for loss value.
    gradients: List of tensors for the gradients of model parameters.
  and nothing to return.
  """

  @abc.abstractmethod
  def __call__(self,
               step: tf.Variable,
               real_particles: TensorLike,
               fantasy_particles: TensorLike,
               loss: tf.Tensor,
               gradients: List[tf.Tensor]):
    pass


class EnergyModel:
  """The energy-model that fits empirical distributions.

  The energy model is a probabilistic model, characterized by an 'energy' E(x)
  of "particles" x, and a "temperature" T. That is, distribution

    q(x) = exp(-E(x)/T) / Z

  where Z is the normalization factor. The E(x) has implicit parameters θ which
  are to be trained.

  Given a emperical distribution p, we want to minimize the KL-divergence
  between p and q. The derivative is derived as

    E_p[∂E/∂θ] - E_q[∂E/∂θ] = (∂/∂θ) (E_p[E] - E_q[E])

  where E_p[f] denotes the expectation of f(x) with x sampled from p.

  To sample from q, we employ stochastic differential equatoins (SDE), based
  on a theorem:

    q is the stationary solution of the Fokker-Planck equation induced by the
    SDE dx = -∇E(x)*dt + dW, with dW ~ Normal(0, 2T*dt).

  To prove this theorem, letting p(x,t) the distribution of an enssemble of
  particles obeying the SDE, we compute (d/dt)KL(p(.,t)|q), where KL indicates
  the Kullback–Leibler divergence. By plugging into the Fokker-Planck equation
  induced by the SDE, which is

    (∂p/∂t)(x,t) = ∇[p(x,t) ∇E(x)] + T Δp(x,t),

  followed by integral by part, we arrive at a negative definite result,
  meaning that KL(p|q) will keep decreasing until p = q. That is, q is the
  stationary solution of the Fokker-Planck equation.

  When the particles are seperated as ambient and latent, say E(x) -> E(v,h)
  where v for ambient and h for latent, then the derivative of KL-divergence
  becomes

    E_p(v)[E_q(h|v)[∂E/∂θ]] - E_q(v,h)[∂E/∂θ]

  where

    q(h|v) := exp(-E(v,h)/T) / ∫dh exp(-E(v,h)/T).

  To sample from q(h|v), we employ a similar SDE, based on a similar theorem:

    q(h|v) is the stationary solution of the Fokker-Planck equation induced by
    the SDE dh = -∇ₕE(v,h)*dt + dW and dv = 0, with dW ~ Normal(0, 2T*dt).

  Implementation Details:
    - All particles are considered as tensors or nested tensors.
    - Attributes `t` and `dt` are non-trainable variables, while `T` is treated
      as constant. It is such designed since `T` shall be fixed for solving the
      energy. While `t` and `dt` can be adjustable.

  Methods:
    __call__: Evolves the data batch.
    get_optimize_fn: Returns a function for step by step training.
  """

  def __init__(self,
               network: tf.Module,
               resample: Callable[[int], TensorLike],
               solver: SDESolver,
               t: ScalarLike,
               T: ScalarLike = None,
               params: List[tf.Variable] = None,
               use_latent: bool = False):
    """
    Args:
      network: The neural network for `x -> -E(x)`, where `E` is the energy.
      resample: The fantasy particles are resampled before sampling by
        evolving SDE. Signature `(batch_size: int) -> particles`, where the
        `particles` is tensor or nested tensor.
      solver: SDE solver.
      t: Time interval of SDE evolution.
      T: The "temperature". Defaults to autmatically determined value.
      params: The parameters. Defaults to the `network.trainable_variables`.
      use_latent: If true, then the network accepts an (ambient, latent) pair
        as inputs, where ambient and latent are tensors or nested tensors.
    """
    self.network = network
    self.resample = resample
    self.solver = solver
    self.t = tf.convert_to_tensor(t, dtype='float32')
    self.params = params if params else network.trainable_variables
    self.use_latent = use_latent

    # x -> -∇E(x)
    def vector_field(x):
      with tf.GradientTape() as tape:
        # Notice that the `tape.watch` and `tape.gradient` can handle nested
        # tensors automatically.
        tape.watch(x)
        # Recall that the network is x -> -E(x).
        y = tf.reduce_sum(self.network(x))
        return tape.gradient(y, x, unconnected_gradients='zero')

    # Determine self.T
    if T is None:
      self.T = get_adaptive_temperature(
          vector_field_samples=vector_field(self.resample(128)),
          t=self.t,
      )
    else:
      self.T = tf.convert_to_tensor(T, dtype='float32')

    @nest_map
    def cholesky(s):
      return tf.sqrt(2 * self.T) * s

    self.sde = SDE(
        vector_field=lambda x, t: vector_field(x),
        cholesky=lambda x, t, s: cholesky(s),
    )
    self.vector_field = vector_field

    # Determine the self.latent_sde.
    if self.use_latent:
      self.latent_sde = SDE(
          vector_field=lambda x, t: dispose_ambient(vector_field(x)),
          cholesky=lambda x, t, s: dispose_ambient(cholesky(s)),
      )

  def __call__(self, batch: TensorLike):
    """Evolves the data batch.

    Args:
      batch: Tensor or nested tensors.

    Returns:
      The evolution result.
      If `self.use_latent`, then the result is the ambient-latent pair.
    """
    particles = self.evolve_real(batch)
    return self.solver(self.sde, 0., self.t, particles)

  def evolve_real(self, batch):
    if not self.use_latent:
      return batch

    # Initialize the latent particles from the resampled fantasy_particles.
    batch_size = tf.shape(batch)[0]
    _, latent = self.resample(batch_size)
    particles = (batch, latent,)

    return self.solver(self.latent_sde, 0., self.t, particles)

  def evolve_fantasy(self, batch_size):
    particles = self.resample(batch_size)
    return self.solver(self.sde, 0., self.t, particles)

  def get_loss(self, real_particles, fantasy_particles):
    # Recall that the network is x -> -E(x), the positions of real and fantasy
    # particles shall be reversed.
    return (
        tf.reduce_mean(self.network(fantasy_particles)) -
        tf.reduce_mean(self.network(real_particles))
    )

  def get_optimize_fn(self,
                      optimizer: keras.optimizers.Optimizer,
                      callbacks: List[Callback] = None):
    """Returns a function for step by step training.

    Args:
      optimizer: An `keras.optimizers.Optimizer` object.
      callbacks: List of `Callback`s.

    Returns:
      A function that accepts an input batch (tensor), then train the model,
      and returns the current training step (int64 scalar).
    """
    callbacks = [] if callbacks is None else callbacks
    step = tf.Variable(0, trainable=False, dtype='int64')

    def train_step(batch: TensorLike):
      real_particles = self.evolve_real(batch)
      batch_size = tf.shape(batch)[0]
      fantasy_particles = self.evolve_fantasy(batch_size)

      with tf.GradientTape() as tape:
        loss = self.get_loss(real_particles, fantasy_particles)
        gradients = tape.gradient(loss, self.params)

      for callback in callbacks:
        callback(step, real_particles, fantasy_particles, loss, gradients)

      optimizer.apply_gradients(zip(gradients, self.params))
      step.assign_add(1)
      return step

    return train_step


def dispose_ambient(x):
  """Sets the ambient component of x to zeros."""
  if isinstance(x, tuple):
    nest_type = lambda *args: args
  elif isinstance(x, dict):
    nest_type = lambda *args: {k: v for k, v in zip(x.keys(), args)}
  else:  # list or namedtuple.
    nest_type = type(x)

  ambient, latent = x
  zeros = map_structure(tf.zeros_like, ambient)
  return nest_type(*[zeros, latent])


def get_adaptive_temperature(
    vector_field_samples: TensorLike,
    t: ScalarLike,
) -> ScalarLike:
  """Suppose that the vector field persists its order during evolution. Thus,
  the proper temperature T shall balance the deterministic and the stochastic
  terms, at least in the starting period of training. That is,

      f(x) t ~ (2T t)^0.5 => T ~ 0.5t f^2(x).

  We use 3-sigma scale as the vector field order. If the vector field is a
  nested tensor, then use the minimum of the orders.
  """
  vector_field_order = minimum(*tf.nest.flatten(map_structure(
      lambda x: 3 * tf.math.reduce_std(x),
      vector_field_samples,
  )))
  return 0.5 * t * vector_field_order**2
