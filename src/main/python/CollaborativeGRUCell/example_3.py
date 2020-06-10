#Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import pathlib
import copy
import json
import six
from petastorm import make_batch_reader

from src.main.python.input.input import Input
from src.main.python.CollaborativeGRUCell.cell import CollaborativeGRUCell


class RNN2RecConfig(object):
    """Configuration for `RNN2Rec Model`."""

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
        """Constructs a `RNN2RecConfig` from a Python dictionary of parameters."""
        config = RNN2RecConfig(user_size=None, item_size=None, chunk_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `RNN2RecConfig` from a json file of parameters."""
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


class CollaborativeRNNModel(tf.keras.Model):
    """RNN2Rec model ("Bidirectional Embedding Representations from a Transformer").

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
        """Constructor for RNN2Rec.

        Args:
        config: `RNN2RecConfig` instance.
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

    def call(self, inputs, training=False):
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

    def fit_(self,
             epochs,
             train_path,
             valid_path,
             shuffle,
             steps_per_epoch,
             validation_steps,
             lr,
             verbose):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided. Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        train_path: Path of parquet file of training data for petastorm execution.
        valid_path: Path of parquet file of validation data for petastorm execution.
        shuffle: Randomly shuffles the elements of this dataset.
       steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            This argument is not supported with array inputs.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If validation_data is a `tf.data` dataset
            and 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted.
        lr: learning rate of the given optimizer
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        """
        self.compile(
            loss=tf.losses.sparse_categorical_crossentropy,
            optimizer=tf.optimizers.RMSprop(lr=lr),
            metrics=['sparse_categorical_accuracy'])

        with make_batch_reader(train_path, num_epochs=None) as reader_train:
            with make_batch_reader(valid_path, num_epochs=None) as reader_validation:

                train_dataset = Input.get_dataset(
                    reader_train,
                    shuffle=shuffle,
                    batch=self.config.batch_size)

                test_dataset = Input.get_dataset(
                    reader_validation,
                    shuffle=shuffle,
                    batch=self.config.batch_size)

                self.fit(
                    train_dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=validation_steps,
                    verbose=verbose)


if __name__ == "__main__":
    train_path = 'file:/' + str(pathlib.Path().absolute()) \
          .replace("\\", "/") \
          .replace("C:/", "//") \
          .replace("/home", "//home") \
          .replace("CollaborativeGRUCell", 'input/data/train/train_data.parquet')

    valid_path = 'file:/' + str(pathlib.Path().absolute()) \
          .replace("\\", "/") \
          .replace("C:/", "//") \
          .replace("/home", "//home") \
          .replace("CollaborativeGRUCell", 'input/data/train/valid_data.parquet')

    config=RNN2RecConfig(user_size=5,item_size=10,batch_size=5, chunk_size=64, num_hidden_layers=128)
    model=CollaborativeRNNModel(config=config)
    model.fit_(
        epochs=100,
        train_path=train_path,
        valid_path=valid_path,
        shuffle=10,
        steps_per_epoch=40,
        validation_steps=10,
        lr=0.001,
        verbose=2)