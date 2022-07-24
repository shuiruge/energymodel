import tensorflow as tf
from .sde import SDE
from .utils import map_structure, nest_map


class Callback:
  """Defines the abstract callback API.

  A callback shall implement the `__call__` method, which has arguments:
    step: An int64 scalar for training step.
    real_particles: Tensor for input batch.
    fantasy_particles: Tensor for input batch.
    loss: Scalar for loss value.
    gradients: List of tensors for the gradients of model parameters.
  and nothing to return.
  """
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
    evolve: Evolves the particles by the SDE.
    get_optimize_fn: Returns a function for step by step training.
  """

  def __init__(self,
               network,
               resample,
               t,
               dt,
               T=None,
               params=None,
               use_latent=False):
    """
    Args:
      network: The neural network for `x -> -E(x)`, where `E` is the energy.
      resample: The fantasy particles are resampled before sampling by
        evolving SDE. Signature `(batch_size: int) -> particles`, where the
        `particles` is tensor or nested tensor. If nested tensor, shall use
        tuple instead of list for nesting.
      t: Time interval of SDE evolution.
      dt: Time step.
      T: The "temperature". Defaults to autmatically determined value.
      params: The parameters. Defaults to the `network.trainable_variables`.
      use_latent: If true, then the network accepts an (ambient, latent) pair
        as inputs, where ambient and latent are tensors or nested tensors.
    """
    self.network = network
    self.resample = resample
    self.t = tf.Variable(t, trainable=False, dtype='float32')
    self.dt = tf.Variable(dt, trainable=False, dtype='float32')
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
    if T is not None:
      self.T = tf.convert_to_tensor(T, dtype='float32')
    else:
      # Suppose that the vector field persists its order during evolution.
      # Thus, the proper T shall balance the deterministic and the stochastic
      # terms, at least in the starting period of training. That is,
      # `f(x) t ~ (2T t)^0.5 => T ~ 0.5t f^2(x)`.
      # We use 3-sigma scale as the vector field order.
      vector_field_order = map_structure(
          lambda x: 3 * tf.math.reduce_std(x),
          vector_field(self.resample(128)),
      )
      # TODO: vector_field_order may be nested, how to compute T from it?
      self.T = 0.5 * t * vector_field_order**2

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

  def evolve(self, particles):
    """Evolves the particles by the SDE.

    Args:
      particles: Tensor or nested tensors.

    Returns:
      The evolution result. The same type as the `particles`.
    """
    return self.sde.evolve(tf.constant(0.), self.t, self.dt, particles)

  def evolve_real(self, batch):
    if not self.use_latent:
      return batch

    # Initialize the latent particles from the resampled fantasy_particles.
    batch_size = tf.shape(batch)[0]
    _, latent = self.resample(batch_size)
    particles = (batch, latent,)

    return self.latent_sde.evolve(tf.constant(0.), self.t, self.dt, particles)

  def evolve_fantasy(self, batch_size):
    particles = self.resample(batch_size)
    return self.sde.evolve(tf.constant(0.), self.t, self.dt, particles)

  def get_loss(self, real_particles, fantasy_particles):
    # Recall that the network is x -> -E(x), the positions of real and fantasy
    # particles shall be reversed.
    return (
        tf.reduce_mean(self.network(fantasy_particles)) -
        tf.reduce_mean(self.network(real_particles))
    )

  def get_optimize_fn(self, optimizer, callbacks=None):
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

    def train_step(batch: tf.Tensor):
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
  ambient, latent = x
  return map_structure(tf.zeros_like, ambient), latent


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
