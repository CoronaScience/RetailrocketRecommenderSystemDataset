import os
import tempfile

from pyspark.sql import SparkSession
from pyspark_config import Config
import tensorflow_datasets as tfds
import pyspark
import tensorflow as tf

conf = Config()
conf.load("input_pipeline.yaml")
conf.apply()

tfrecord_location = 'Output/'
name = "test.tfrecord"
filename = os.path.join(tfrecord_location, name)

train_dataset = tf.data.TFRecordDataset(
    filename
)

print(train_dataset)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

for x,y in train_dataset:
    print("output: ", x,y)


