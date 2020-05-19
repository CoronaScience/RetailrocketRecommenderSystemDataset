import tempfile

from pyspark_config import Config
from pathlib import Path
from pyspark_config.output import *
from pyspark_config.input import *
from pyspark_config.spark_utils.dataframe_extended import DataFrame_Extended
from pyspark_config.transformations.transformations import *
from pyspark.sql import SparkSession



print("Reading path...")
conf = Config()
conf.load(Path("resource/config_test.yaml"))
print("Path readed...")
print(conf)

print(tempfile.gettempdir())

conf.apply()