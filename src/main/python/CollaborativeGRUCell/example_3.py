#@title Licensed under the Apache License, Version 2.0 (the "License");
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

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import argparse
import numpy as np
import os
import pathlib
from petastorm import make_batch_reader
from src.main.python.input.input import Input

from src.main.python.CollaborativeGRUCell.cell import CollaborativeGRUCell

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5,
                        help="number of sequences processed in parallel")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="number of unrolled steps in BPTT")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="number of hidden units in the RNN cell")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="RMSprop learning rate")
    parser.add_argument("--max-train-chunks", type=int, default=None,
                        help="max number of chunks per user for training")
    parser.add_argument("--max-valid-chunks", type=int, default=None,
                        help="max number of chunks per user for validation")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="number of epochs to run")
    parser.add_argument("--rho", type=float, default=0.9,
                        help="RMSprop decay coefficient")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="enable display of debugging messages")
    return parser.parse_args()

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
batch_size= 5
chunk_size= 64
num_users= 5
num_items= 10
args = _parse_args()
EPOCHS= 100
settings = {
    "chunk_size": args.chunk_size,
    "batch_size": args.batch_size,
    "hidden_size": args.hidden_size,
    "learning_rate": args.learning_rate,
    "rho": args.rho,
}

class CollaborativeRNNModel(tf.keras.Model):

    def __init__(self, num_users, num_items, is_training, seq_length, chunk_size=128, batch_size=1, hidden_size=128,
                 learning_rate=0.1, rho=0.9, *args, **kwargs):
        super(CollaborativeRNNModel, self).__init__(*args, **kwargs)
        self._batch_size = batch_size
        self.num_items = num_items
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.rho = rho
        self.chunk_size = chunk_size
        self.is_training = is_training
        # RNN cell.
        cell = CollaborativeGRUCell(hidden_size, num_users, num_items)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.states = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self.ws = self.add_weight("weight", [self.hidden_size, self.num_items + 1],
                             dtype=tf.float32, trainable=True)

    def call(self, inputs, training=False):
        # Compute the states and final state for each element of the batch.
        states, final_states = self.states(inputs, initial_state=self._initial_state)

        # Output layer.
        # `output` has shape (batch_size * chunk_size, hidden_size).
        output = tf.reshape(tf.concat(states, axis=1), [-1, self.hidden_size])

        # `logits` has shape (batch_size * chunk_size, num_items).
        logits = tf.matmul(output, self.ws)
        return logits

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def batch_size(self):
        return self._batch_size

    @initial_state.setter
    def initial_state(self, value):
        self._initial_state = value


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



with make_batch_reader(train_path, num_epochs=None) as reader_train:
    with make_batch_reader(valid_path, num_epochs=None) as reader_validation:
        train_dataset = Input.get_dataset(reader_train, shuffle=10, batch=5)
        test_dataset = Input.get_dataset(reader_validation, shuffle=10, batch=5)

        model = CollaborativeRNNModel(num_users, num_items, is_training=True, seq_length=64, **settings)
        model.compile(loss=tf.losses.sparse_categorical_crossentropy, optimizer=tf.optimizers.RMSprop(lr=0.001),
                      metrics=['sparse_categorical_accuracy'])

        model.fit(train_dataset, epochs=100, steps_per_epoch=40, validation_data=test_dataset, validation_steps=10,verbose=2)
