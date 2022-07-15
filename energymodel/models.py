import tensorflow as tf
from typing import Callable, List
from abc import ABC, abstractmethod
from .utils import nabla, RandomWalk, clip_value


class Callback(ABC):
  """Defines the abstract callback API."""

  @abstractmethod
  def __call__(
      self,
      step: tf.Tensor,
      batch: tf.Tensor,
      loss: tf.Tensor,
      gradients: List[tf.Tensor],
  ):
    return NotImplemented


class EnergyModel:
  """The energy-model that fits empirical distributions."""

  def __init__(
      self,
      network: tf.Module,
      fantasy_particles: tf.Tensor,
      resample: Callable[[tf.Tensor], tf.Tensor],
      clip_gradient: float = None,
  ):
    """Initialize the energy-model.

    Args:
        network (tf.Module): The function x -> -E(x), where E is the energy.
        fantasy_particles (tf.Tensor): The initial value of fantasy particles.
        resample (Callable[[tf.Tensor], tf.Tensor]): The fantasy particles are
          resampled before sampling by random walk. For example,

              resample = lambda x: random_uniform(tf.shape(x))
        
          will randomly reset the fantasy particles before random walk.
        clip_gradient (float, optional): Clipe the ∇E, just for safty.
          Defaults to None.
    """
    self.network = network
    self.fantasy_particles = tf.Variable(
        fantasy_particles, trainable=False, dtype='float32')
    self.resample = resample
    assert clip_gradient is None or clip_gradient > 0
    self.clip_gradient = clip_gradient

    self.params = self.network.trainable_variables
    
    # x -> -∇E(x)
    self.vector_field = nabla(self.network)
    if self.clip_gradient is not None:
      self.vector_field = clip_value(
          self.vector_field,
          -self.clip_gradient,
          self.clip_gradient,
      )
  
  def __call__(self, x: tf.Tensor, random_walk: RandomWalk):
    return random_walk.evolve(self.vector_field, x)

  def evolve(self, random_walk: RandomWalk):
    self.fantasy_particles.assign(
        self.resample(self.fantasy_particles)
    )
    random_walk.inplace_evolve(self.vector_field, self.fantasy_particles)

  def get_loss(self, real_particles: tf.Tensor):
    # Recall that the network is x -> -E(x), the positions of real and fantasy
    # particles shall be reversed.
    return (
        tf.reduce_mean(self.network(self.fantasy_particles)) -
        tf.reduce_mean(self.network(real_particles))
    )
  
  def get_optimize_fn(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        random_walk: RandomWalk,
        callbacks: List[Callback] = None,
    ):
    callbacks = [] if callbacks is None else callbacks
    step = tf.Variable(0, trainable=False, dtype='int64')

    def train_step(batch: tf.Tensor):
      # Evolve the fantasy particles
      self.evolve(random_walk)

      # Compute loss gradients
      with tf.GradientTape() as tape:
        loss = self.get_loss(batch)
        gradients = tape.gradient(loss, self.params)

      for callback in callbacks:
        callback(step, batch, loss, gradients)

      # Optimize
      optimizer.apply_gradients(zip(gradients, self.params))

      step.assign_add(1)
      return step

    return train_step
  