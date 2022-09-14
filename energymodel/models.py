import abc
import tensorflow as tf
from typing import Callable, List
from .sde import SDE, SDESolver
from .utils import TensorLike, map_structure, nest_map, minimum


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

    q(x) is the stationary solution of the Fokker-Planck equation induced by
    the SDE dx = -∇E(x)*dt + dW, with dW ~ Normal(0, 2T*dt).

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
               t: float,
               T: float = None,
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
                      optimizer: tf.optimizers.Optimizer,
                      callbacks: List[Callback] = None):
    """Returns a function for step by step training.

    Args:
      optimizer: An `tf.optimizers.Optimizer` object.
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
    t: float,
) -> float:
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


class LossMonitor(Callback):

  def __init__(self,
               writer: tf.summary.SummaryWriter,
               log_steps: int):
    self.writer = writer
    self.log_steps = tf.constant(int(log_steps), dtype='int64')

  def __call__(self, step, real_particles, fantasy_particles, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        tf.summary.scalar('loss', loss, step)


class FantasyParticleMonitor(Callback):

  def __init__(self,
               writer: tf.summary.SummaryWriter,
               model: EnergyModel,
               log_steps: int):
    self.writer = writer
    self.model = model
    self.log_steps = tf.constant(int(log_steps), dtype='int64')

  def __call__(self, step, real_particles, fantasy_particles, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        map_structure(
            lambda x, step: tf.summary.histogram('fantasy_particles', x, step),
            fantasy_particles,
            step,
        )


class VectorFieldMonitor(Callback):

  def __init__(self,
               writer: tf.summary.SummaryWriter,
               model: EnergyModel,
               log_steps: int):
    self.writer = writer
    self.model = model
    self.log_steps = tf.constant(int(log_steps), dtype='int64')

  def __call__(self, step, real_particles, fantasy_particles, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        map_structure(
            lambda x, step: tf.summary.histogram(
                'vector_field/real_particles', x, step),
            self.model.vector_field(real_particles),
            step,
        )
        map_structure(
            lambda x, step: tf.summary.histogram(
                'vector_field/fantasy_particles', x, step),
            self.model.vector_field(fantasy_particles),
            step,
        )


class LossGradientMonitor(Callback):

  def __init__(self,
               writer: tf.summary.SummaryWriter,
               model: EnergyModel,
               log_steps: int):
    self.writer = writer
    self.model = model
    self.log_steps = tf.constant(int(log_steps), dtype='int64')

  def __call__(self, step, real_particles, fantasy_particles, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        for var, grad in zip(self.model.params, gradients):
          # TensorBoard cannot log the gradients directly, so we hack it by
          # making copies of the gradients, via `tf.identity`.
          tf.summary.histogram(
              'gradient/' + var.name,
              tf.identity(grad),
              step,
          )
