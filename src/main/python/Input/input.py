import os
import pathlib

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark_config import Config
import tensorflow as tf


class Input(object):
    @staticmethod
    def config_load(path):
        conf = Config()
        conf.load(path)
        conf.apply()

    @staticmethod
    def get_dataset(path, shuffle, batch):
        path_='file:/'+path

        def make_window_dataset(ds, window_size=64, shift=64, stride=1):
            windows = ds.window(window_size, shift=shift, stride=stride)

            def sub_to_batch(sub):
                return sub.batch(window_size, drop_remainder=True)

            windows = windows.flat_map(sub_to_batch)
            return windows

        def create_dataset(tensor):
            return tf.data.Dataset.from_tensor_slices(tensor)

        with make_batch_reader(path_, num_epochs=1) as reader:
            train_dataset = make_petastorm_dataset(reader).map(lambda x: x.item_list).unbatch()
            windows = train_dataset.map(lambda x: make_window_dataset(create_dataset(x))).flat_map(lambda x: x)

            for x in windows:
                print(x)

print("\nIf you mean the current working directory:\n")
print(str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//").replace("/home","///home")+"/Output/test.parquet")



petastorm_dataset_url = str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//").replace("/home","//home")+"/Output/test.parquet"

#Input.config_load("input_pipeline.yaml")
Input.get_dataset(petastorm_dataset_url, shuffle=100, batch=2)

