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
import tqdm

from image_utils.image_preprocess import get_shifted_image, get_zooming_image, get_rotated_image
from CSNet.csnet import get_pretrained_CSNet

# best crop dataset
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
    
# unlabeled dataset(Open Images)
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
        return transformed_image, bounding_box, perturbated_bounding_box, suggestion_label, adjustment_label, magnitude_label

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
        adjustment_label = [0.0] * len(adjustment_magnitude_list)
        adjustment_label[index] = 1

        for mag in magnitude_list:
            magnitude_label = [0.0] * len(adjustment_magnitude_list)
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
            'suggestion': 1.0,
            'adjustment': best_adjustment_label[1],
            'magnitude': best_adjustment_label[2]
        }
    else:
        return {
            'name': image_name,
            'suggestion': 0.0,
            'adjustment': [0.0] * len(adjustment_magnitude_list),
            'magnitude': [0.0] * len(adjustment_magnitude_list)
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

def perturbating(image, bounding_box, func):
    output = None
    for i in range(0, 100000):
        if func == 0:
            output = get_shifted_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test', direction=0)
        elif func == 1:
            output = get_shifted_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test', direction=1)
        elif func == 3:
            output = get_rotated_image(image, bounding_box, allow_zero_pixel=False, option='vapnet_test')
        if output != None:
            break
    if output == None:
        return None
    perturbated_image, operator, new_box = output
    if func != 3:
        new_box = [
            [new_box[0], new_box[1]],
            [new_box[0], new_box[3]],
            [new_box[2], new_box[3]],
            [new_box[2], new_box[1]],
        ]
    adjustment = [0.0] * 6
    magnitude = [0.0] * 6
    if operator[func] < 0:
        if func == 3:
            adjustment_index = (func - 1) * 2
        else:
            adjustment_index = func * 2
    else:
        if func == 3:
            adjustment_index = (func - 1) * 2 + 1
        else:
            adjustment_index = func * 2 + 1
    adjustment[adjustment_index] = 1.0
    magnitude[adjustment_index] = -operator[func]
    return perturbated_image, new_box, adjustment, magnitude

def make_annotation_for_labeled(image_path, bounding_box):
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1].split('.')[0]

    best_crop = image.crop(bounding_box)
    best_crop.save(os.path.join('./data/image_labeled_vapnet', image_name + f'_-1.jpg'))
    annotation_list = []
    perturbated_image_cnt = [0, 0, 0]
    box_corners = [
        [bounding_box[0], bounding_box[1]],
        [bounding_box[0], bounding_box[3]],
        [bounding_box[2], bounding_box[3]],
        [bounding_box[2], bounding_box[1]],   
    ]
    func_index = [0, 1, 3]
    i = 0
    while i < len(func_index):
        func = func_index[i]
        output = perturbating(image, bounding_box, func)
        if output == None:
            i += 1
            continue
        perturbated_image = output[0]
        new_box = output[1]
        adjustment_label = output[2]
        magnitude_label = output[3]
        perturbated_image_name = image_name + f'_{func}_{perturbated_image_cnt[i]}.jpg'
        annotation = {
            'name': perturbated_image_name,
            'bounding_box': box_corners,
            'perturbated_bounding_box': new_box,
            'suggestion': [1.0],
            'adjustment': adjustment_label,
            'magnitude': magnitude_label
        }
        """
        loop_cnt = 0
        flag = False
        for prev_annotation in annotation_list:
            perturbated_bounding_box = prev_annotation['perturbated_bounding_box']
            if perturbated_bounding_box == new_box:
                flag = True
                loop_cnt += 1
                if loop_cnt == 10:
                    i += 1
                    loop_cnt = 0
                    break
                break
        if flag:
            continue
        """
        perturbated_image.save(os.path.join('./data/image_labeled_vapnet', perturbated_image_name))
        annotation_list.append(annotation)
        if perturbated_image_cnt[i] < 3:
            perturbated_image_cnt[i] += 1
            i -= 1
        i += 1

    annotation_list.append({
        'name': image_name + f'_-1.jpg',
        'bounding_box': box_corners,
        'perturbated_bounding_box': box_corners,
        'suggestion': [0.0],
        'adjustment': [0.0] * 6,
        'magnitude': [0.0] * 6
    })
    return annotation_list

def make_annotations_for_labeled(data_list, image_dir_path):
    annotation_list = []
    for data in tqdm.tqdm(data_list):
        image_name = data['name']
        bounding_box = data['crop']
        image_path = os.path.join(image_dir_path, image_name)
        annotation_list_one_image = make_annotation_for_labeled(image_path, bounding_box)
        annotation_list += annotation_list_one_image
    with open('./data/annotation/labeled_vapnet/labeled_testing_set.json', 'w') as f:
        json.dump(annotation_list, f, indent=2)
    return
                
if __name__ == '__main__':
    cfg = Config()
    data_list = []
    
    with open('./data/annotation/best_crop/best_testing_set.json', 'r') as f:
        data_list = json.load(f)
    
    make_annotations_for_labeled(data_list, './data/image')
    
    """
    with open('./data/annotation/labeled_vapnet/labeled_testing_set.json', 'r') as f:
        data_list = json.load(f)

    cnt = [0, 0, 0, 0, 0, 0, 0]
    for data in data_list:
        suggestion = data['suggestion']
        adjustment = data['adjustment']
        if suggestion == [0.0]:
            cnt[6] += 1
        else:
            print(data)
            cnt[adjustment.index(1.0)] += 1
    print(cnt)
    """
