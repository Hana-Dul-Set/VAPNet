import os
import math

class Config:
    def __init__(self):
        
        self.image_dir = '../data/image'

        self.data_dir = '../data/annotation'
        self.weight_dir = './output/weight'

        self.scored_crops_data = os.path.join(self.data_dir, 'scored_crops')
        self.best_crop_data = os.path.join(self.data_dir, 'best_crop')
        self.unlabeled_data = os.path.join(self.data_dir, 'unlabeled')

        self.gpu_id = 0
        self.num_workers = 1

        self.pairwise_margin = 0.2
        self.learning_rate = 2 * math.exp(-5)
        self.weight_decay = 5 * math.exp(-4)

        self.max_epoch = 100

        self.scored_crops_batch_size = 1
        self.scored_crops_N = 7
        self.best_crop_K = 8
        self.unlabeled_P = 8

        self.image_size = (299, 299)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]