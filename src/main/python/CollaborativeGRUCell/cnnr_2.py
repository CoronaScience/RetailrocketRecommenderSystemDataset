from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import six.moves
import tensorflow as tf
import time
from math import sqrt
import os
import pathlib
from petastorm import make_batch_reader
from src.main.python.input.input import Input

from src.main.python.CollaborativeGRUCell.cell import CollaborativeGRUCell
from src.main.python.CollaborativeGRUCell.reader import Dataset

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

    def call(self, inputs, training=False):
        # Compute the states and final state for each element of the batch.
        states, final_states = self.states(inputs, initial_state=self._initial_state)

        # Output layer.
        # `output` has shape (batch_size * chunk_size, hidden_size).
        output = tf.reshape(tf.concat(states, axis=1), [-1, self.hidden_size])
        with tf.compat.v1.variable_scope("output"):
            ws = tf.compat.v1.get_variable("weights", [self.hidden_size, self.num_items + 1],
                                 dtype=tf.float32)

        # `logits` has shape (batch_size * chunk_size, num_items).
        logits = tf.matmul(output, ws)
        return final_states, logits

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def batch_size(self):
        return self._batch_size

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

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

with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)

  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    masked = per_example_loss * tf.compat.v1.to_float(tf.sign(labels))
    masked = tf.reshape(masked, [batch_size, chunk_size])
    return tf.reduce_sum(masked, axis=1)

  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          name='test_accuracy')

  model = CollaborativeRNNModel(num_users, num_items, is_training=True, seq_length=64, **settings)

  optimizer = tf.keras.optimizers.Adam()

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  def train_step(inputs):
      input, labels = inputs

      with tf.GradientTape() as tape:
          predictions = model(input)
          loss = compute_loss(labels, predictions)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_accuracy.update_state(labels, predictions)
      return loss


  def test_step(inputs):
      images, labels = inputs

      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)

      test_loss.update_state(t_loss)
      test_accuracy.update_state(labels, predictions)


  # `run` replicates the provided computation and runs it
  # with the distributed input.
  @tf.function
  def distributed_train_step(dataset_inputs):
      per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)


  @tf.function
  def distributed_test_step(dataset_inputs):
      return strategy.run(test_step, args=(dataset_inputs,))


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

  with make_batch_reader(train_path, num_epochs=1) as reader_train:
      with make_batch_reader(valid_path, num_epochs=1) as reader_validation:
          train_dataset = Input.get_dataset(reader_train, shuffle=100, batch=5)
          test_dataset = Input.get_dataset(reader_validation, shuffle=100, batch=5)

          train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
          test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

          for epoch in range(EPOCHS):
              # TRAIN LOOP
              total_loss = 0.0
              num_batches = 0
              for x in train_dist_dataset:
                  total_loss += distributed_train_step(x)
                  num_batches += 1
              train_loss = total_loss / num_batches

              # TEST LOOP
              for x in test_dist_dataset:
                  distributed_test_step(x)

              if epoch % 2 == 0:
                  checkpoint.save(checkpoint_prefix)

              template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                          "Test Accuracy: {}")
              print(template.format(epoch + 1, train_loss,
                                    train_accuracy.result() * 100, test_loss.result(),
                                    test_accuracy.result() * 100))

              test_loss.reset_states()
              train_accuracy.reset_states()
              test_accuracy.reset_states()