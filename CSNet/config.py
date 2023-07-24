import os

class Config:
    def __init__(self, mode='train'):
        
        self.image_dir = '../data/image'

        self.data_dir = '../data/annotation'

        self.scored_crops_data = os.path.join(self.data_dir, 'scored_crops')
        self.best_crop_data = os.path.join(self.data_dir, 'best_crop')
        self.unlabeled_data = os.path.join(self.data_dir, 'unlabeled')

        self.scored_crops_N = 16
        self.best_crop_K = 16
        self.unlabeled_P = 16

        self.image_size = (229, 229)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]