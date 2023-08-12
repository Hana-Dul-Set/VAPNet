import os
import math

import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import tqdm

from CSNet.image_utils.image_preprocess import get_shifted_image, get_zooming_image, get_rotated_image
from CSNet.csnet import get_pretrained_CSNet
from config import Config

# best crop dataset for training(FCDB, GAICD)
class BCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.best_crop_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'best_training_set.json')

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
    
# unlabeled dataset for training(Open Images)
class UnlabledDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unlabeled_training_set.json')

        self.data_list = self.build_data_list()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['name']
        image = Image.open(os.path.join(self.image_dir, image_name))
        suggestion_label = data['suggestion']
        adjustment_label = data['adjustment']
        magnitude_label = data['magnitude']
        return image, suggestion_label, adjustment_label, magnitude_label

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        return data_list
    
# Labeled dataset for test(FCDB, GAICD)
class LabledDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = os.path.join(self.cfg.image_dir, 'image_labeled_vapnet')
        self.dataset_path = self.cfg.labeled_data
        
        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'labeled_testing_set.json')

        self.data_list = self.build_data_list()

        self.transformer = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['name']
        image = Image.open(os.path.join(self.image_dir, image_name))
        image_size = torch.tensor(image.size)
        if len(image.getbands()) == 1:
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image, (0, 0, image.width, image.height))
            image = rgb_image
        transformed_image = self.transformer(image)
        bounding_box = torch.tensor(data['bounding_box'])
        perturbated_bounding_box = torch.tensor(data['perturbated_bounding_box'])
        suggestion_label = torch.tensor(data['suggestion'])
        adjustment_label = torch.tensor(data['adjustment'])
        magnitude_label = torch.tensor(data['magnitude'])
        return transformed_image, image_size, bounding_box, perturbated_bounding_box, suggestion_label, adjustment_label, magnitude_label

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        return data_list