from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import os
from tqdm import tqdm
import time
from PIL import Image
from torchvision.transforms import transforms
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score as f1
from shapely.geometry import Polygon

from config import Config
from vapnet import VAPNet
from dataset import LabledDataset
from image_utils.image_preprocess import get_shifted_box, get_rotated_box

def build_dataloader(cfg):
    labeled_dataset = LabledDataset('test', cfg)
    data_loader = DataLoader(dataset=labeled_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers)
    return data_loader

class Tester(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = os.path.join(self.cfg.image_dir, 'image_labeled_vapnet')

        self.data_loader = build_dataloader(self.cfg)
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.batch_size = self.cfg.batch_size

        self.adjustment_count = self.cfg.adjustment_count

        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')

        self.data_length = self.data_loader.__len__()

        self.fpr_limit = self.cfg.fpr_limit

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.auc_score_sum = 0
        self.tpr_score_sum = 0
        self.f1_score_sum = [0] * (self.adjustment_count)
        self.iou_score_sum = 0

    def run(self):
        self.model.eval().to(self.device)
        # self.model.eval()
        print('\n======test start======\n')
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.data_loader), total=self.data_length):
                if index == 1:
                    break
                # data split
                image = data[0].to(self.device)
                gt_bounding_box = data[1].tolist()
                gt_perturbated_bounding_box = data[2].tolist()
                gt_suggesiton_label = data[3].numpy()
                gt_adjustment_label = data[4].numpy()
                gt_magnitude_label = data[5].numpy()

                # model inference
                predicted_suggestion, predicted_adjustment, predicted_magnitude = self.model(image)

                # calculate auc, tpr, and threshold for suggestion
                auc_score, tpr_score, threshold = self.calculate_suggestion_accuracy(gt_suggesiton_label, predicted_suggestion)
                
                # remove no-suggested elements
                predicted_suggestion = predicted_suggestion.to('cpu').numpy()
                predicted_adjustment = predicted_adjustment.to('cpu').numpy()
                predicted_magnitude = predicted_magnitude.to('cpu').numpy()

                suggested_index = np.where(predicted_suggestion >= threshold)[0]

                filtered_gt_adjustment_label = gt_adjustment_label[suggested_index]
                filtered_predicted_adjustment = predicted_adjustment[suggested_index]

                # calculate f1 score for each adjustment
                f1_score = list(self.calculate_f1_score(filtered_gt_adjustment_label, filtered_predicted_adjustment))

                # get one-hot encoded predicted adjustment
                one_hot_predicted_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=predicted_adjustment)

                # remove no-suggestd elements
                suggested_index = list(suggested_index)
                
                # get predicted bounding box
                predicted_bounding_box = []

                for index, gt_perturbated_box in enumerate(gt_perturbated_bounding_box):
                    if index not in suggested_index:
                        predicted_bounding_box.append(gt_perturbated_box)
                        continue
                
                    adjustment = np.where(one_hot_predicted_adjustment[index] == 1.0)[0][0]
                    magnitude = predicted_magnitude[index][adjustment]
                    #  horizontal shift
                    if adjustment == 0 or adjustment == 1:
                        predicted_box = get_shifted_box(image=image[index], \
                                                        bounding_box_corners=gt_perturbated_box, \
                                                        mag=magnitude,\
                                                        direction=0)
                    # vertical shift
                    elif adjustment == 2 or adjustment == 3:
                        predicted_box = get_shifted_box(image=image[index], \
                                                        bounding_box_corners=gt_perturbated_box, \
                                                        mag=magnitude,\
                                                        direction=1)
                    # rotation
                    elif adjustment == 4 or adjustment == 5:
                        if gt_adjustment_label[index][4] == 1 or gt_adjustment_label[index][5] == 1:
                            predicted_box = get_rotated_box(bounding_box=gt_bounding_box[index], \
                                                            input_radian=magnitude - gt_magnitude_label[index][adjustment])
                        else:
                            predicted_box = get_rotated_box(bounding_box=gt_perturbated_box, \
                                                            input_radian=magnitude)
                    predicted_bounding_box.append(predicted_box)

                # calculate average iou score for each bounding box pairs
                iou_score = self.calculate_ave_iou_score(gt_bounding_box, predicted_bounding_box)
                
                # add each score
                self.auc_score_sum += auc_score
                self.tpr_score_sum += tpr_score
                self.f1_score_sum = [x + y for x, y in zip(f1_score, self.f1_score_sum)]
                self.iou_score_sum += iou_score

        print('\n======test end======\n')

        # calculate ave score
        print("auc_sum:", self.auc_score_sum)
        print("tpr_sum:", self.tpr_score_sum)
        ave_auc_score = self.auc_score_sum / self.data_length
        ave_tpr_score = self.tpr_score_sum / self.data_length
        ave_f1_score = [x / self.data_length for x in self.f1_score_sum]
        ave_iou_score = self.iou_score_sum / self.data_length

        accuracy_log = f'{ave_auc_score:.5f}/{ave_tpr_score:.5f}/{ave_f1_score}/{ave_iou_score:.5f}'
        print(accuracy_log)

    def calculate_suggestion_accuracy(self, gt_suggestion, predicted_suggestion):
        def find_idx_for_fpr(fpr):
            idx = (np.abs(fpr - self.fpr_limit)).argmin()
            return idx
        gt_suggestion = gt_suggestion
        predicted_suggestion = predicted_suggestion.to('cpu')
        fpr, tpr, cut = roc_curve(gt_suggestion, predicted_suggestion)
        auc_score = auc(fpr, tpr)
        idx = find_idx_for_fpr(fpr)

        tpr_score = tpr[idx]
        threshold = cut[idx]
        print('gt suggestion:', gt_suggestion)
        print('predicted_suggestion:', predicted_suggestion)
        print('FPR:', fpr)
        print('TPR:', tpr)
        print('CUT:', cut)
        print(f'idx: {idx}/threshold: {threshold}')
        return auc_score, tpr_score, threshold
    
    def convert_array_to_one_hot_encoded(self, array):
        largest_value = np.max(array)
        one_hot_encoded = np.zeros_like(array)
        one_hot_encoded[array == largest_value] = 1
        return one_hot_encoded
    
    def calculate_f1_score(self, gt_adjustment, predicted_adjustment):
        def convert_one_hot_encoded_to_index(array):
            if np.all(array == 0):
                return np.array(self.adjustment_count + 1)
            one_index = np.where(array == 1)[0][0]
            return one_index

        print('gt adjustment:', gt_adjustment)
        print('predicted adjustment:', predicted_adjustment)
        one_hot_encoded_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, axis=1, arr=predicted_adjustment)
        print('predicted adjustment:', one_hot_encoded_adjustment)
        gt_label = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=gt_adjustment)
        predicted_label = np.apply_along_axis(convert_one_hot_encoded_to_index, axis=1, arr=one_hot_encoded_adjustment)
        print('gt label:', gt_label)
        print('predicted label:', predicted_label)
        labels = [i for i in range(0, self.adjustment_count + 1)]
        f1_score = f1(gt_label, predicted_label, labels=labels, average=None, zero_division=0.0)
        f1_score = f1_score[:-1]
        print('f1 score:', f1_score)
        return f1_score
        
    def calculate_ave_iou_score(self, boudning_box_list, perturbated_box_list):
        # box format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (counter-clockwise order)
        def calculate_iou_score(box1, box2):
            print("gt_box:", box1, "/predcited_box:", box2)
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area

            iou = intersection_area / union_area if union_area > 0 else 0.0
            return iou
        
        iou_sum = 0
        for i in range(len(boudning_box_list)):
            iou_sum += calculate_iou_score(boudning_box_list[i], perturbated_box_list[i])
        
        ave_iou = iou_sum / len(boudning_box_list)
        return ave_iou

def test_while_training():
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()