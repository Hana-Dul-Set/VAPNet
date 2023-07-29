from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import os
import json
from config import Config
import random
import math
from torchvision.transforms import transforms
import torch

from image_utils.image_preprocess import get_shifted_image, get_zooming_image, get_rotated_image
from CSNet.csnet import get_pretrained_CSNet

# best crop dataset
class LabeledDataset(Dataset):
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
    
# unlabeled dataset(Open Images)
class UnlabledDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.image_dir = self.cfg.image_dir
        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unlabeled_training_set.json')

        self.data_list = self.build_data_list()
        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['name']
        image = Image.open(os.path.join(self.image_dir, image_name))
        image = self.transformer(image)
        suggestion_label = data['suggestion']
        adjustment_label = data['adjustment']
        magnitude_label = data['magnitude']
        return image, suggestion_label, adjustment_label, magnitude_label

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        return data_list
    
def get_csnet_score(image_list, csnet):
    # device = None
    transformer = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ])
    tensor = []
    for image in image_list:
        # Grayscale to RGB
        if len(image.getbands()) == 1:
            rgb_image = Image.new("RGB", image.size)
            rgb_image.paste(image, (0, 0, image.width, image.height))
            image = rgb_image
        tensor.append(transformer(image))
    tensor = torch.stack(tensor, dim=0)
    score_list = csnet(tensor)
    return score_list
    
def make_pseudo_label(image_path):
    
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1]

    left_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    right_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]
    up_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    down_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]

    """
    zoom_in_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    zoom_out_magnitude = [x * 0.05 for x in range(1, 10, 1)]
    """

    clockwise_magnitude = [-x * math.pi / 36 for x in range(1, 10, 1)]
    counter_clokwise_magnitude = [x * math.pi / 36 for x in range(1, 10, 1)]

    adjustment_magnitude_list = [left_shift_magnitude,
                                 right_shift_magnitude,
                                 up_shift_magnitude,
                                 down_shift_magnitude,
                                 # zoom_in_magnitude,
                                 # zoom_out_magnitude,
                                 clockwise_magnitude,
                                 counter_clokwise_magnitude]
    
    csnet = get_pretrained_CSNet()
    original_image_score = get_csnet_score([image], csnet).item()

    pseudo_data_list = []
    
    for index, magnitude_list in enumerate(adjustment_magnitude_list):
        adjustment_label = [0] * len(adjustment_magnitude_list)
        adjustment_label[index] = 1

        for mag in magnitude_list:
            magnitude_label = [0] * len(adjustment_magnitude_list)
            pseudo_image = None

            # Get vertical shifted images
            if 0 <= index < 2:
                pseudo_image = get_shifted_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 mag=mag,
                                                 direction=0)
                magnitude_label[index] = mag
            # Get horiziontal shifted images
            elif 2 <= index < 4:
                pseudo_image = get_shifted_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 mag=mag,
                                                 direction=1)
                magnitude_label[index] = mag
            # Get clockwise and counter clockwise rotated images
            elif 4 <= index < 6:
                pseudo_image = get_rotated_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 radian=mag)
                magnitude_label[index] = mag
            
            score = get_csnet_score([pseudo_image], csnet)[0].item()
            pseudo_data_list.append((score, adjustment_label, magnitude_label))

    pseudo_data_list.sort(reverse=True)

    print(len(pseudo_data_list), original_image_score)

    best_adjustment_label = pseudo_data_list[0]
    best_adjustment_score = best_adjustment_label[0]
    if original_image_score + 0.2 < best_adjustment_score:
        return {
            'name': image_name,
            'suggestion': 1,
            'adjustment': best_adjustment_label[1],
            'magnitude': best_adjustment_label[2]
        }
    else:
        return {
            'name': image_name,
            'suggestion': 0,
            'adjustment': [0] * len(adjustment_magnitude_list),
            'magnitude': [0] * len(adjustment_magnitude_list)
        }
    
def make_annotations_for_unlabeled(image_list, image_dir_path):
    annotation_list = []
    for image_name in image_list:
        image_path = os.path.join(image_dir_path, image_name)
        annotation = make_pseudo_label(image_path)
        annotation_list.append(annotation)
    with open('./data/annotation/unlabeled_vapnet/unlabeled_training_set.json', 'w') as f:
        json.dump(annotation_list, f, indent=2)
    return
                
if __name__ == '__main__':
    cfg = Config()
    """
    ret = make_pseudo_label('./data/sample.jpg')
    print(ret)
    """
    ret = UnlabledDataset('train', cfg).__getitem__(0)
    print(ret)