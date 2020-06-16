import pathlib
import json

from recommendator.CollaborativeGRUCell.model import CollaborativeRNNModel, CollaborativeRNN2RecConfig
from recommendator.CollaborativeGRUCell.training import fit_

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
fit_(model=model,
     epochs=100,
     train_path=train_path,
     valid_path=valid_path,
     shuffle=10,
     steps_per_epoch=40,
     validation_steps=10,
     lr=0.001,
     verbose=2)