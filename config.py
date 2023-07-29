import os
import math

class Config:
    def __init__(self):
        
        self.image_dir = './data/image'

        self.data_dir = './data/annotation'
        self.weight_dir = './output/weight'

        self.unlabeled_data = os.path.join(self.data_dir, 'unlabeled_vapnet')

        self.gpu_id = 0
        self.num_workers = 1

        self.learning_rate = 2e-5
        self.weight_decay = 5e-4

        self.max_epoch = 100

        self.image_size = (299, 299)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]