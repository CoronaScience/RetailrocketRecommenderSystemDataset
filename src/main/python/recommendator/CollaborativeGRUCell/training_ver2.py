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

from recommendator.input.input import Input
from recommendator.CollaborativeGRUCell.model import CollaborativeRNNModel, CollaborativeRNN2RecConfig


def input_fn(filenames, shuffle, batch):
    raw_dataset = tf.data.TFRecordDataset(filenames=filenames)

    feature_description = {
        'user_list': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
        'item_list': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True)
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = raw_dataset.map(_parse_function)

    def make_window_dataset(ds, window_size=64, shift=64, stride=1):
        windows = ds.window(window_size, shift=shift, stride=stride)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        windows = windows.flat_map(sub_to_batch)
        return windows

    def create_dataset(features, label):
        features_ = tf.data.Dataset.from_tensor_slices(tf.reshape(features, shape=[128, 2]))
        label_ = tf.concat([label, tf.constant([0.])], 0)
        return (make_window_dataset(features_),
                make_window_dataset(tf.data.Dataset.from_tensor_slices(label_[1:])))

    def func(x, y):
        return tf.data.Dataset.zip((x, y))

    def mapping(x):
        return (tf.stack([x['user_list'], x['item_list']], 1), x['item_list'])

    def reshape(x, y):
        return (x, tf.cast(tf.reshape(y, [-1]), tf.int32))

    features = dataset.map(mapping)
    features_windows = features.map(create_dataset).flat_map(func)
    return features_windows.shuffle(shuffle).batch(batch).map(reshape)


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
    return optimizer


@tf.function
def train_step(model, optimizer, features, targets):
    # keep track of our gradients

    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the
		# loss
        pred = model(features)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    # calculate the gradients using our tape and then update the
    # model weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def training(epochs, num_of_iters, optimizer, train_dataset):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    epoch_steps = 0

    for epoch in range(epochs):
        for iteration, (input, target) in enumerate(train_dataset):
            # Break if the number of computed batches exceeds the
            # total number of the examples
            if (iteration % num_of_iters)==0 and iteration >0:
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
    'batch_size': 2,
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
    'lr' : 0.004,
    'verbose' : 2
}

config=CollaborativeRNN2RecConfig.from_dict(config_data)
model=CollaborativeRNNModel(config=config)
optimizer=create_optimizer(
        init_lr=0.002,
        num_train_steps=3000)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

filename="/home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/recommendator/input/data/train/train_data.tfrecord/part-r-00000"

df=input_fn(filenames=filename, shuffle=10, batch=2)

training(epochs=300, num_of_iters=4, optimizer=optimizer, train_dataset=df)
