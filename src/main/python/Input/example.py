from pyspark.sql import SparkSession
from pyspark_config import Config
#import tensorflow_datasets as tfds
import pyspark
import numpy as np
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from pyspark.sql.types import IntegerType
from petastorm import make_reader

HelloWorldSchema = Unischema('HelloWorldSchema', [
   UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
   UnischemaField('other_data', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
])


def row_generator(x):
   """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
   return {'id': x,
           'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
           'other_data': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}


def generate_hello_world_dataset(output_url='file:///home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/example'):
   rows_count = 10
   rowgroup_size_mb = 256

   spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
   sc = spark.sparkContext

   # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
   # well as save petastorm specific metadata
   with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

       rows_rdd = sc.parallelize(range(rows_count))\
           .map(row_generator)\
           .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

       spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
           .coalesce(10) \
           .write \
           .mode('overwrite') \
           .parquet(output_url)

generate_hello_world_dataset()

with make_reader('file:///home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/example') as reader:
    dataset = make_petastorm_dataset(reader)
    iterator = dataset.make_one_shot_iterator()
    tensor = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sample = sess.run(tensor)
        print(sample)

