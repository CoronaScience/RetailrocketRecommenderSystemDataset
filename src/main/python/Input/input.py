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
            for x in train_dataset:
                print(x)

petastorm_dataset_url = "//home/patrizio/PycharmProjects/RetailrocketRecommenderSystemDataset/src/main/python/Input/Output/test.parquet"

#Input.config_load("input_pipeline.yaml")
Input.get_dataset(petastorm_dataset_url, shuffle=100, batch=2)

