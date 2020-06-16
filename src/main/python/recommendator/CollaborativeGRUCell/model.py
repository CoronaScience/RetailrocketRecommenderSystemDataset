import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_USER_WEIGHT_VARIABLE_NAME = "users"
_ITEMS_WEIGHTS_VARIABLE_NAME = "items"

import six
import json
import copy
from tensorflow.python.ops.rnn_cell_impl import *


class CollaborativeRNN2RecConfig(object):
    """Configuration for `CollaborativeRNN2Rec Model`."""

    def __init__(self,
                 user_size,
                 item_size,
                 chunk_size,
                 batch_size=1,
                 num_hidden_layers=12):

        """Constructs RNN2RecConfig.
        Args:
        user_size: Vocabulary size of `user_ids`.
        item_size: Vocabulary size of `item_ids`.
        chunk_size: Length of sequence
        batch_size: Size of batches in training.
        num_hidden_layers: Number of hidden layers in the RNN sequence.
        """
        self.user_size = user_size
        self.item_size = item_size
        self.chunk_size = chunk_size
        self.batch_size= batch_size
        self.num_hidden_layers = num_hidden_layers

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `CollaborativeRNN2RecConfig` from a Python dictionary of parameters."""
        config = CollaborativeRNN2RecConfig(user_size=None, item_size=None, chunk_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `CollaborativeRNN2RecConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CollaborativeGRUCell(LayerRNNCell):
  """Gated Recurrent Unit cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnGRU` for better performance on GPU, or
  `tf.contrib.rnn.GRUBlockCellV2` for better performance on CPU.

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().

      References:
    Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation:
      [Cho et al., 2014]
      (https://aclanthology.coli.uni-saarland.de/papers/D14-1179/d14-1179)
      ([pdf](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf))
  """

  def __init__(self,
               num_units,
               num_users,
               num_items,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               **kwargs):
    super(CollaborativeGRUCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    _check_supported_dtypes(self.dtype)

    if tf.executing_eagerly() and context.num_gpus() > 0:
      logging.warn(
          "%s: Note that this cell is not optimized for performance. "
          "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
          "performance on GPU.", self)
    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    self._num_users = num_users
    self._num_items = num_items
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
    _check_supported_dtypes(self.dtype)
    input_depth = inputs_shape[-1]
    self._gate_kernel_users = self.add_weight(
        "gates/%s" % _USER_WEIGHT_VARIABLE_NAME,
        shape=[self._num_users + 1, self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_kernel_items = self.add_weight(
        "gates/%s" % _ITEMS_WEIGHTS_VARIABLE_NAME,
        shape=[self._num_items + 1, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_weight(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel_users = self.add_variable(
        "candidate/%s" % _USER_WEIGHT_VARIABLE_NAME,
        shape=[self._num_users + 1, self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_kernel_items = self.add_variable(
        "candidate/%s" % _ITEMS_WEIGHTS_VARIABLE_NAME,
        shape=[self._num_items + 1, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_weight(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    _check_rnn_cell_input_dtypes([inputs, state])

    w_hidden_u = tf.nn.embedding_lookup(self._gate_kernel_users, tf.cast(inputs[:, 0], tf.int32))
    w_input_i = tf.nn.embedding_lookup(self._gate_kernel_items, tf.cast(inputs[:, 1], tf.int32))
    res_h = tf.matmul(tf.expand_dims(state, 1), w_hidden_u)

    gate_inputs = nn_ops.bias_add(res_h, self._gate_bias)

    value = math_ops.sigmoid((tf.squeeze(gate_inputs, [1]) + w_input_i))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    w_hidden_u = tf.nn.embedding_lookup(self._candidate_kernel_users, tf.cast(inputs[:, 0], tf.int32))
    w_input_i = tf.nn.embedding_lookup(self._candidate_kernel_items, tf.cast(inputs[:, 1], tf.int32))

    res_h = tf.matmul(tf.expand_dims(state, 1), w_hidden_u)
    candidate = nn_ops.bias_add(res_h, self._candidate_bias)

    c = self._activation(tf.squeeze(candidate, [1]) + w_input_i)
    new_h = u * state + (1 - u) * c
    return new_h, new_h

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "kernel_initializer": initializers.serialize(self._kernel_initializer),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(CollaborativeGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _check_supported_dtypes(dtype):
  if dtype is None:
    return
  dtype = dtypes.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex or dtype.is_integer):
    raise ValueError("RNN cell only supports floating point inputs, "
                     "but saw dtype: %s" % dtype)


def _check_rnn_cell_input_dtypes(inputs):
  """Check whether the input tensors are with supported dtypes.

  Default RNN cells only support floats and complex as its dtypes since the
  activation function (tanh and sigmoid) only allow those types. This function
  will throw a proper error message if the inputs is not in a supported type.

  Args:
    inputs: tensor or nested structure of tensors that are feed to RNN cell as
      input or state.

  Raises:
    ValueError: if any of the input tensor are not having dtypes of float or
      complex.
  """
  for t in nest.flatten(inputs):
    _check_supported_dtypes(t.dtype)


class CollaborativeRNNModel(tf.keras.Model):
    """CollaborativeRNN model ("Collaborative Recurrent Neural Networks for Dynamic Recommender Systems").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
    label_embeddings = tf.get_variable(...)
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        """Constructor for CollaborativeRNNModel.

        Args:
        config: `CollaborativeRNN2RecConfig` instance.
        is_training: bool. rue for training model, false for eval model. Controls
                    whether dropout will be applied.
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
            it is must faster if this is True, on the CPU or GPU, it is faster if
            this is False.
        scope: (optional) variable scope. Defaults to "bert".
        Raises:
        ValueError: The config is invalid or one of the input tensor shapes is invalid.
        """

        super(CollaborativeRNNModel, self).__init__(*args, **kwargs)
        self.config= config
        self.cell = CollaborativeGRUCell(
            num_units=config.num_hidden_layers,
            num_users=config.user_size,
            num_items=config.item_size)
        self._initial_state = self.cell.zero_state(
            batch_size=config.batch_size,
            dtype=tf.float32)
        self.states = tf.keras.layers.RNN(
            self.cell,
            return_state=True,
            return_sequences=True)
        self.ws = self.add_weight(
            name="weight",
            shape=[config.num_hidden_layers, config.item_size + 1],
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs, training=True):
        # Compute the states and final state for each element of the batch.
        states, final_states = self.states(inputs, initial_state=self._initial_state)

        # Output layer.
        # `output` has shape (batch_size * chunk_size, hidden_size).
        output = tf.reshape(tf.concat(states, axis=1), [-1, self.config.num_hidden_layers])

        # `logits` has shape (batch_size * chunk_size, num_items).
        logits = tf.matmul(output, self.ws)
        return logits

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        self._initial_state = value