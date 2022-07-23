import tensorflow as tf


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

  return tf.nest.map_structure(map_fn, *nest_args, **kwargs)


def is_nest_structure(x):
  return isinstance(x, (list, tuple))


def nest_map(fn):
  """A convenient decorator for applying `map_structure`, converting a normal
  function to the function that accepts nested inputs.

  Examples:
  >>> fn = lambda x, y: x + y
  >>> x = [tf.constant(1.), tf.constant(2.)]  # x is nested.
  >>> y = tf.constant(3.)  # y is not nested.
  >>> nest_map(fn)(x, y)  # => [(1+3), (2+3)].
  """

  def decorated(*args, **kwargs):
    return map_structure(fn, *args, **kwargs)

  return decorated
