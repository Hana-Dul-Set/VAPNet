from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import os
import json
from config import Config
import random

from image_preprocess import get_cropping_image, get_zooming_out_image, get_shifted_image, get_rotated_image
from augmentation import shift_borders, zoom_out_borders, rotation_borders


# scored crops dataset
class SCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.scored_crops_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'crops_training_set.json')

        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'crops_testing_set.json')

        self.image_list, self.data_list = self.build_data_list()

        self.random_crops_count = self.cfg.scored_crops_N

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        crops_list = self.data_list[index]
        selected_crops_list = random.sample(crops_list, self.random_crops_count)
        return image_name, selected_crops_list

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        crops_list = []
        for data in data_list:
            image_list.append(data['name'])
            crops_list.append(data['crops'])
        return image_list, crops_list

# best crop dataset
class BCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.best_crop_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'best_training_set.json')

        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'best_testing_set.json')

        self.image_list, self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        best_crop_bounding_box = self.data_list[index]
        return image_name, best_crop_bounding_box

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        best_crop_list = []
        for data in data_list:
            image_list.append(data['name'])
            best_crop_list.append(data['crop'])
        return image_list, best_crop_list
    
# unlabeled dataset
class UNDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unsplash_training_set_20230726_1608.json')

        self.image_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        return image_name

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        image_list = []
        for data in data_list:
            image_list.append(data['name'])
        return image_list

if __name__ == '__main__':
    cfg = Config()

    idx = 0
    dataset = SCDataset('train', cfg)
    # make_pairs_scored_crops(dataset.__getitem__(0))
    dataset = UNDataset('train', cfg)
    print(dataset.__getitem__(idx))    