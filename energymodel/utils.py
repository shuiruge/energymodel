import tensorflow as tf


def minimum(*args):
  """Extends the `tf.math.minimum` to arbitrarily many arguments."""
  if len(args) < 2:
    return args[0]
  elif len(args) == 2:
    return tf.math.minimum(*args)
  else:
    return minimum(minimum(args[0], args[1]), *args[2:])


def maximum(*args):
  """Extends the `tf.math.maximum` to arbitrarily many arguments."""
  if len(args) < 2:
    return args[0]
  elif len(args) == 2:
    return tf.math.maximum(*args)
  else:
    return maximum(maximum(args[0], args[1]), *args[2:])


def random_uniform(size):
  return tf.random.uniform(
      shape=size, minval=-1.0, maxval=1.0, dtype='float32')


def map_structure(fn, *args, **kwargs):
  """Analogy to the `tf.nest.map_structure`, but not all `args` are nested."""
  nest_args = [arg for arg in args if is_nest_structure(arg)]
  if not nest_args:
    return fn(*args, **kwargs)

  def map_fn(*nest_args, **kwargs):
    all_args = []
    offset = 0
    for arg in args:
      if is_nest_structure(arg):
        all_args.append(nest_args[offset])
        offset += 1
      else:
        all_args.append(arg)
    return fn(*all_args, **kwargs)

  # `tf.nest.map_structure` will assert same structure automatically.
  return tf.nest.map_structure(map_fn, *nest_args, **kwargs)


def is_nest_structure(x):
  """Auxiliary function of `map_structure`."""
  return isinstance(x, (list, tuple))


def nest_map(fn):
  """A convenient decorator for applying `map_structure`, converting a normal
  function to the function that accepts nested inputs.

  Examples:
  >>> fn = lambda x, y: x + y
  >>> x = [[tf.constant(0.), tf.constant(1.)], tf.constant(2.)]  # x is nested.
  >>> y = tf.constant(3.)  # y is not nested.
  >>> nest_map(fn)(x, y)  # => [[(0+3), (1+3)], (2+3)].
  """

  def decorated(*args, **kwargs):
    return map_structure(fn, *args, **kwargs)

  return decorated
