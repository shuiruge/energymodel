import abc
import tensorflow as tf
from typing import List
from .utils import nabla
from .sde import SDE


class Callback(abc.ABC):
  """Defines the abstract callback API."""

  @abc.abstractmethod
  def __call__(self,
               step: tf.Tensor,
               batch: tf.Tensor,
               loss: tf.Tensor,
               gradients: List[tf.Tensor]):
    return NotImplemented


class EnergyModel:
  """The energy-model that fits empirical distributions.

  Notes:
    - Attributes `t` and `dt` are non-trainable variables, while `T` is treated
      as constant.
  """

  def __init__(self,
               network,
               fantasy_particles,
               resample,
               t,
               dt,
               T=None,
               params=None):
    """
    Args:
      network: The neural network for x -> -E(x), where E is the energy.
      fantasy_particles: The initial value of fantasy particles.
      resample: The fantasy particles are resampled before sampling by
        evolving SDE. For example,

            resample = lambda x: random_uniform(tf.shape(x))

        will randomly reset the fantasy particles before evolving SDE.
      t: Time interval of SDE evolving.
      dt: Time step.
      T: The "temperature". Defaults to autmatically determined value.
      params: The parameters. Defaults to the `network.trainable_variables`.
    """
    self.network = network
    self.fantasy_particles = tf.Variable(
        fantasy_particles, trainable=False, dtype='float32')
    self.resample = resample
    self.t = tf.Variable(t, trainable=False, dtype='float32')
    self.dt = tf.Variable(dt, trainable=False, dtype='float32')
    self.params = params if params else network.trainable_variables

    # x -> -∇E(x)
    self.vector_field = nabla(self.network)

    if T is not None:
      self.T = tf.convert_to_tensor(T, dtype='float32')
    else:
      # Suppose that the vector field persists its order during evolution.
      # Thus, the proper T shall balance the deterministic and the stochastic
      # terms, at least in the starting period of training. That is,
      # `f(x) t ~ (2T t)^0.5 => T ~ 0.5t f^2(x)`.
      # We use 2-sigma scale as the vector field order.
      vector_field_order = 3 * tf.math.reduce_std(
          self.vector_field(self.resample(self.fantasy_particles))
      )
      self.T = 0.5 * t * vector_field_order**2

    self.sde = SDE(
        vector_field=lambda x, t: self.vector_field(x),
        cholesky=lambda x, t, s: tf.sqrt(2 * self.T) * s,
    )

  def evolve(self, x):
    t0 = tf.constant(0.)
    return self.sde.evolve(t0, self.t, self.dt, x)

  def evolve_fantasy(self):
    self.fantasy_particles.assign(
        self.evolve(self.resample(self.fantasy_particles))
    )

  def get_loss(self, real_particles):
    # Recall that the network is x -> -E(x), the positions of real and fantasy
    # particles shall be reversed.
    return (
        tf.reduce_mean(self.network(self.fantasy_particles)) -
        tf.reduce_mean(self.network(real_particles))
    )

  def get_optimize_fn(self, optimizer, callbacks=None):
    callbacks = [] if callbacks is None else callbacks
    step = tf.Variable(0, trainable=False, dtype='int64')

    def train_step(batch: tf.Tensor):
      self.evolve_fantasy()

      with tf.GradientTape() as tape:
        loss = self.get_loss(batch)
        gradients = tape.gradient(loss, self.params)

      for callback in callbacks:
        callback(step, batch, loss, gradients)

      optimizer.apply_gradients(zip(gradients, self.params))
      step.assign_add(1)
      return step

    return train_step


class LossMonitor(Callback):

  def __init__(self,
               writer: tf.summary.SummaryWriter,
               log_steps: int):
    self.writer = writer
    self.log_steps = tf.constant(int(log_steps), dtype='int64')

  def __call__(self, step, batch, loss, gradients):
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

  def __call__(self, step, batch, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        tf.summary.histogram(
            'fantasy_particles',
            self.model.fantasy_particles,
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

  def __call__(self, step, batch, loss, gradients):
    if tf.equal(step % self.log_steps, 0):
      with self.writer.as_default():
        tf.summary.histogram(
            'vector_field/real_particles',
            self.model.vector_field(batch),
            step,
        )
        tf.summary.histogram(
            'vector_field/fantasy_particles',
            self.model.vector_field(self.model.fantasy_particles),
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

  def __call__(self, step, batch, loss, gradients):
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
