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
from petastorm import make_batch_reader
import pathlib

import warnings
warnings.filterwarnings("ignore")

from recommendator.input.input import Input
from recommendator.CollaborativeGRUCell.model import CollaborativeRNNModel, CollaborativeRNN2RecConfig


def fit_(model,
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
        model.compile(
            loss=tf.losses.sparse_categorical_crossentropy,
            optimizer=tf.optimizers.RMSprop(lr=lr),
            metrics=['sparse_categorical_accuracy'])

        with make_batch_reader(train_path, num_epochs=None) as reader_train:
            with make_batch_reader(valid_path, num_epochs=None) as reader_validation:

                train_dataset = Input.get_dataset(
                    reader_train,
                    shuffle=shuffle,
                    batch=model.config.batch_size)

                test_dataset = Input.get_dataset(
                    reader_validation,
                    shuffle=shuffle,
                    batch=model.config.batch_size)

                model.fit(
                    train_dataset,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=validation_steps,
                    verbose=verbose)


def create_optimizer(
        init_lr,
        num_train_steps,
        num_warmup_steps=None):
    """Creates an optimizer training op."""
    global_step = tf.Variable(1, name="global_step")

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.compat.v1.train.polynomial_decay(
        learning_rate=learning_rate,
        global_step= global_step,
        decay_steps= num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

    new_global_step = global_step + 1
    return optimizer


@tf.function
def train_step(model, optimizer, features, targets):
    # keep track of our gradients

    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the
		# loss
        pred = model(features)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, pred)

    # calculate the gradients using our tape and then update the
    # model weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def training(epochs):
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        with make_batch_reader(train_path, num_epochs=5) as reader_train:
            train_dataset = Input.get_dataset(
                reader_train,
                shuffle=10,
                batch=model.config.batch_size)

            # Training loop - using batches of 32
            for x, y in train_dataset:
                # Optimize the model
                loss_value = train_step(model, optimizer, x, y)

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(y, model(x, training=True))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", epoch)

            if epoch % 2 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))


def training2(epochs, num_of_iters):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    with make_batch_reader(train_path, num_epochs=None) as reader_train:
        train_dataset = Input.get_dataset(
            reader_train,
            shuffle=10,
            batch=model.config.batch_size)

        epoch_steps = 0

        for epoch in range(epochs):
            for iteration, (input, target) in enumerate(train_dataset):
                # Break if the number of computed batches exceeds the
                # total number of the examples
                if iteration % num_of_iters==0 and iteration >0:
                    epoch_steps+=1
                    train_loss_results.append(epoch_loss_avg.result())
                    train_accuracy_results.append(epoch_accuracy.result())
                    if epoch_steps % 2 == 0:
                        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch_steps,
                                                                                    epoch_loss_avg.result(),
                                                                                    epoch_accuracy.result()))
                    break
                # HERE WE PERFORM ONE TRAINING STEP

                loss_value = train_step(model, optimizer, input, target)

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(target, model(input, training=True))



config_data = {
    'user_size': 5,
    'item_size': 10,
    'batch_size': 5,
    'chunk_size': 64,
    'num_hidden_layers': 128
}

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

train_config_data={
    'epochs' : 100,
    'train_path' : train_path,
    'valid_path' : valid_path,
    'shuffle' : 10,
    'steps_per_epoch' : 40,
    'validation_steps' : 10,
    'lr' : 0.001,
    'verbose' : 2
}

config=CollaborativeRNN2RecConfig.from_dict(config_data)
model=CollaborativeRNNModel(config=config)
optimizer=create_optimizer(
        init_lr=0.004,
        num_train_steps=100)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []


training2(100, 40)



