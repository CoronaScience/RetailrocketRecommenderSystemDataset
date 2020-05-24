import os
import tempfile

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
from petastorm.codecs import ScalarCodec
from pyspark.sql.types import IntegerType
#conf = Config()
#conf.load("input_pipeline.yaml")
#conf.apply()

spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()

df= spark.read.parquet("/home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/test2.parquet")
df.show(10)

output_url="file:///home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/withschema.parquet"

FrameSchema = Unischema('FrameSchema',
[
 UnischemaField('user', np.int32, (), ScalarCodec(IntegerType()), nullable=False),
 UnischemaField('item', np.int32, (), ScalarCodec(IntegerType()), nullable=False),
 UnischemaField('date', np.int32, (), ScalarCodec(IntegerType()), nullable=False),
])

def row_generator(x):
 return {'user': np.asarray(1),'item': np.asarray(1), 'date': np.asarray(1)}

#with materialize_dataset(spark, output_url, FrameSchema):
# rows_rdd = spark.sparkContext.parallelize(range(128*5))\
# .map(row_generator)\
# .map(lambda x: dict_to_spark_row(FrameSchema, x))
# spark.createDataFrame(rows_rdd, FrameSchema.as_spark_schema()) \
# .write \
# .parquet(output_url)

petastorm_dataset_url = "file:///home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/test.parquet"
petastorm_dataset_url2 = "file:///home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/withschema.parquet"
with make_batch_reader(petastorm_dataset_url, num_epochs=1) as reader:
  train_dataset = make_petastorm_dataset(reader).map(lambda x: (x.user, x.item_list)).unbatch().shuffle(100).batch(1)
  for x, y in train_dataset:
      print(x, y)


