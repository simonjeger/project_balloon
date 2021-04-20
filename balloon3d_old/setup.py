from generate_wind_map import generate_wind_map
from autoencoder_train import autoencoder_train

from pathlib import Path

# Build folder structure if it doesn't exist yet
Path('weights_autoencoder').mkdir(parents=True, exist_ok=True)
Path('weights_model').mkdir(parents=True, exist_ok=True)
Path('data').mkdir(parents=True, exist_ok=True)
Path('data/test').mkdir(parents=True, exist_ok=True)
Path('data/test/image').mkdir(parents=True, exist_ok=True)
Path('data/test/tensor').mkdir(parents=True, exist_ok=True)
Path('data/test/tensor_comp').mkdir(parents=True, exist_ok=True)
Path('data/train').mkdir(parents=True, exist_ok=True)
Path('data/train/image').mkdir(parents=True, exist_ok=True)
Path('data/train/tensor').mkdir(parents=True, exist_ok=True)
Path('data/train/tensor_comp').mkdir(parents=True, exist_ok=True)

size_x = 30
size_y = 30
size_z = 10

generate_wind_map(size_x, size_y, size_z, 10, 'train')
generate_wind_map(size_x, size_y, size_z, 10, 'test')

autoencoder_train()

#import model_train
