import os
import math

import json
from PIL import Image
import torch
from torchvision.transforms import transforms
import tqdm

from CSNet.image_utils.image_preprocess import get_shifted_image, get_zooming_image, get_rotated_image
from CSNet.csnet import get_pretrained_CSNet
from config import Config

def get_csnet_score(image_list, csnet, device):
    # device = None
    image_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformer = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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
    tensor = tensor.to(device)
    score_list = csnet(tensor)
    return score_list
    
def make_pseudo_label(image_path):
    
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1]

    left_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    right_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]
    up_shift_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    down_shift_magnitude = [x * 0.05 for x in range(1, 10, 1)]

    # zoom_in_magnitude = [-x * 0.05 for x in range(1, 10, 1)]
    # zoom_out_magnitude = [x * 0.05 for x in range(1, 10, 1)]

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
    
    device = 'cpu'
    csnet = get_pretrained_CSNet(device)
    pseudo_data_list = []
    
    for index, magnitude_list in enumerate(adjustment_magnitude_list):
        adjustment_label = [0.0] * len(adjustment_magnitude_list)
        adjustment_label[index] = 1

        for mag in magnitude_list:
            magnitude_label = [0.0] * len(adjustment_magnitude_list)
            pseudo_image = None

            # get vertical shifted images
            if 0 <= index < 2:
                pseudo_image = get_shifted_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 mag=mag,
                                                 direction=0)
                magnitude_label[index] = mag
            # get horiziontal shifted images
            elif 2 <= index < 4:
                pseudo_image = get_shifted_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 mag=mag,
                                                 direction=1)
                magnitude_label[index] = mag
            # get clockwise and counter clockwise rotated images
            elif 4 <= index < 6:
                pseudo_image = get_rotated_image(image,
                                                 [0, 0, image.size[0], image.size[1]],
                                                 allow_zero_pixel=True,
                                                 option='vapnet',
                                                 input_radian=mag)
                magnitude_label[index] = mag
            
            # get csnet score of each perturbed image
            score = get_csnet_score([pseudo_image], csnet, device)[0].item()
            pseudo_data_list.append((score, adjustment_label, magnitude_label))

    # sort in desceding order by csnet score
    pseudo_data_list.sort(reverse=True)

    original_image_score = get_csnet_score([image], csnet, device).item()
    best_adjustment_label = pseudo_data_list[0]
    best_adjustment_score = best_adjustment_label[0]

    print("pseudo_data_list:", pseudo_data_list)
    print("length:", len(pseudo_data_list))
    print("original_image_score:", original_image_score)

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

def perturbing_for_labeled_data(image, bounding_box, func):
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
    perturbed_image, operator, new_box = output
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

    return perturbed_image, new_box, adjustment, magnitude

def make_annotation_for_labeled(image_path, bounding_box):
    image = Image.open(image_path)
    image_name = image_path.split('/')[-1].split('.')[0]

    best_crop = image.crop(bounding_box)
    best_crop.save(os.path.join('./data/image/image_labeled_vapnet', image_name + f'_-1.jpg'))
    annotation_list = []
    perturbed_image_cnt = [0, 0, 0]
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
        output = perturbing_for_labeled_data(image, bounding_box, func)
        if output == None:
            i += 1
            continue
        perturbed_image = output[0]
        new_box = output[1]
        adjustment_label = output[2]
        magnitude_label = output[3]
        perturbed_image_name = image_name + f'_{func}_{perturbed_image_cnt[i]}.jpg'
        annotation = {
            'name': perturbed_image_name,
            'bounding_box': box_corners,
            'perturbed_bounding_box': new_box,
            'suggestion': [1.0],
            'adjustment': adjustment_label,
            'magnitude': magnitude_label
        }
        perturbed_image.save(os.path.join('./data/image/image_labeled_vapnet', perturbed_image_name))
        annotation_list.append(annotation)
        if perturbed_image_cnt[i] < 4:
            perturbed_image_cnt[i] += 1
            i -= 1
        i += 1

    annotation_list.append({
        'name': image_name + f'_-1.jpg',
        'bounding_box': box_corners,
        'perturbed_bounding_box': box_corners,
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

def count_images_by_perturbation(annotation_path):
    # count the pseuo label images by adjustment
    with open(annotation_path, 'r') as f:
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
    perturbed_image_sum = sum(cnt) - cnt[6]
    print(perturbed_image_sum)
    print(cnt)
    return

def remove_duplicated_box(annotation_path):
    data_list = []
    new_data_list = []
    with open(annotation_path, 'r') as f:
        data_list = json.load(f)
    for data in data_list:
        new_box = data['perturbed_bounding_box']
        bounding_box = data['bounding_box']
        flag = False
        for new_data in new_data_list:
            if new_data['bounding_box'] == bounding_box and new_data['perturbed_bounding_box'] == new_box:
                flag = True
                break
        if flag == False:
            new_data_list.append(data)
    with open(annotation_path, 'w') as f:
        json.dump(new_data_list, f, indent=2)

if __name__ == '__main__':
    cfg = Config()
    data_list = []

    labeled_annotation_path = './data/annotation/best_crop/best_testing_set.json'
    
    print(make_pseudo_label('./data/sample.jpg'))
    """
    with open('./data/annotation/best_crop/best_testing_set.json', 'r') as f:
        data_list = json.load(f)
    make_annotations_for_labeled(data_list, './data/image')
    """