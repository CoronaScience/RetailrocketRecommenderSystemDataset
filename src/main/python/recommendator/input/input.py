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
    def get_dataset(reader, shuffle, batch):
        def make_window_dataset(ds, window_size=64, shift=64, stride=1):
            windows = ds.window(window_size, shift=shift, stride=stride)

            def sub_to_batch(sub):
                return sub.batch(window_size, drop_remainder=True)

            windows = windows.flat_map(sub_to_batch)
            return windows

        def create_dataset(features, label):
            features_= tf.data.Dataset.from_tensor_slices(tf.reshape(features, shape=[128, 2]))
            label_ = tf.concat([label, tf.constant([0.])], 0)
            return (make_window_dataset(features_),
                    make_window_dataset(tf.data.Dataset.from_tensor_slices(label_)))

        def func(x,y):
            return tf.data.Dataset.zip((x,y))

        def mapping(x):
            return (tf.stack([x.user_list, x.item_list], 2) , x.item_list)

        def reshape(x, y):
            return (x, tf.cast(tf.reshape(y, [-1]), tf.int32))

        features = make_petastorm_dataset(reader).map(mapping).unbatch()
        features_windows = features.map(create_dataset).flat_map(func)
        return features_windows.shuffle(shuffle).batch(batch).map(reshape)



print("\nIf you mean the current working directory:\n")
print(str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//").replace("/home","///home")+"/Output/test.parquet")

petastorm_dataset_url = str(pathlib.Path().absolute()).replace("\\","/").replace("C:/","//").replace("/home","//home")+"/Output/test.parquet"

#input.config_load("valid_data_config.yaml")
#path_= 'file:/'+petastorm_dataset_url
#with make_batch_reader(path_, num_epochs=1) as reader:
#    ds=Input.get_dataset(reader, shuffle=100, batch=1)

#    for x, y in ds:
#        print("input:",x, '\n')
#        print("output:", y, '\n')
