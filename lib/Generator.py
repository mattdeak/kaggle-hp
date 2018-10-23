import numpy as np
from numpy import random
import matplotlib.image as im
import os


class DataGenerator:
    """A generator for reading the files of the Protein Atlas
    kaggle competition. Reads in batches determines by `batch_size`.
    
    Raises:
        StopIteration -- Raised after all files have been returned
    
    Returns:
        np.ndarray -- An array of images in dimensions 
                      (# of Samples, Height, Width, # of Channels)
    """
    def __init__(self, data_directory, batch_size, channels=['green','red','yellow','blue']):
        self.dir = data_directory
        self.channels = channels
        self.index = 0
        self.unique_ids = list(
            set([file.split('_')[0] for file in os.listdir(data_directory)])
        )
        self.n = len(self.unique_ids)
        self.batch_size = batch_size
        random.shuffle(self.unique_ids)

    def reset(self, shuffle=True):
        self.index = 0
        if shuffle:
            random.shuffle(self.unique_ids)
        
    def _get_image(self, image_id, channel):
        return im.imread(f'{self.dir}/{image_id}_{channel}.png')
    
    def __next__(self):
        if self.index > self.n:
            raise StopIteration
            
        batch_size = min(self.batch_size, self.n - self.index)
        ids = self.unique_ids[self.index:self.index + batch_size]
        self.index += batch_size
        
        images = [[self._get_image(image_id, channel) for channel in self.channels] for image_id in ids]
        yield np.array(images).reshape(-1, 512, 512, len(self.channels))
    
