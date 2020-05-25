import os
import pathlib

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark_config import Config


class Input(object):
    @staticmethod
    def config_load(path):
        conf = Config()
        conf.load(path)
        conf.apply()

    @staticmethod
    def get_dataset(path, shuffle, batch):
        path_='file:/'+path
        with make_batch_reader(path_, num_epochs=1) as reader:
            train_dataset = make_petastorm_dataset(reader).map(lambda x: x.item_list).unbatch()

            feature_length = 10
            label_length = 5

            range_ds = tf.data.Dataset.range(100000)

            def dense_1_step(batch):
                # Shift features and labels one step relative to each other.
                return batch[:-1], batch[1:]

            predict_dense_1_step = train_dataset.map(dense_1_step)

            for features, label in predict_dense_1_step.take(3):
                print(features.numpy(), " => ", label.numpy())

            for x in train_dataset:
                print(x)

print("\nIf you mean the current working directory:\n")
print(str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//")+"/Output/test.parquet")



petastorm_dataset_url = str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//")+"/Output/test.parquet"

#Input.config_load("input_pipeline.yaml")
Input.get_dataset(petastorm_dataset_url, shuffle=100, batch=2)

