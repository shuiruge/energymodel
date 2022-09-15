import tensorflow as tf

from .models import Callback, EnergyModel
from .utils import map_structure, check_nan


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


# TODO: Add docstring.
class NanMonitor(Callback):

  def __init__(self, check_steps: int):
    self.check_steps = tf.constant(int(check_steps), dtype='int64')

  def __call__(self, step, real_particles, fantasy_particles, loss, gradients):
    if tf.equal(step % self.check_steps, 0):
      check_nan(real_particles, 'real_particles')
      check_nan(fantasy_particles, 'fantasy_particles')
      check_nan(loss, 'loss')
