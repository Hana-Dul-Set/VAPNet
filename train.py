from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import os
import time
from torchvision.transforms import transforms

from config import Config
from vapnet import VAPNet
from dataset import BCDataset, UnlabledDataset
from image_utils.augmentation import *
from image_utils.image_preprocess import get_cropping_image, get_zooming_image, get_shifted_image, get_rotated_image

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    bc_dataset = BCDataset('train', cfg)
    labeled_loader = DataLoader(dataset=bc_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    unlabeled_dataset = UnlabledDataset('train', cfg)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    return labeled_loader, unlabeled_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir
        
        self.adjustment_count = self.cfg.adjustment_count

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = self.cfg.batch_size
        
        self.train_iter = 0

        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')
        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)
        
        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.suggestion_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.magnitude_loss_sum = 0
        self.suggested_case_iter = 0

    def training(self):
        self.model.train().to(self.device)
        # self.model.train()
        self.bc_loader, self.unlabeled_loader = build_dataloader(self.cfg)
        print('\n======train start======\n')

        for index, data in enumerate(zip(self.bc_loader, self.unlabeled_loader)):
            self.train_iter += 1

            bc_data_list = data[0]
            unlabeled_data_list = data[1]

            # get randomly perturbated image and label for suggestion case
            l_image_list, l_suggestion_label_list, l_adjustment_label_list, l_magnitude_label_list = self.get_labeled_data_list(bc_data_list)
            # get best crop image and label for no-suggestion case
            b_image_list, b_suggestion_label_list, b_adjustment_label_list, b_magnitude_label_list = self.get_best_crop_labeled_data_list(bc_data_list)

            # combine 
            l_image_list += b_image_list
            l_suggestion_label_list += b_suggestion_label_list
            l_adjustment_label_list += b_adjustment_label_list
            l_magnitude_label_list += b_magnitude_label_list

            # get unlabeled data label
            ul_image_list = [x[0] for x in unlabeled_data_list]
            ul_suggestion_label_list = [x[1] for x in unlabeled_data_list]
            ul_adjustment_label_list = [x[2] for x in unlabeled_data_list]
            ul_magnitude_label_list = [x[3] for x in unlabeled_data_list]

            # combine
            image_list = l_image_list + ul_image_list
            suggestion_list = l_suggestion_label_list + ul_suggestion_label_list
            adjustment_list = l_adjustment_label_list + ul_adjustment_label_list
            magnitude_list = l_magnitude_label_list + ul_magnitude_label_list

            # shuffle
            combined_list = list(zip(image_list, suggestion_list, adjustment_list, magnitude_list))
            random.shuffle(combined_list)
            image_list, suggestion_list, adjustment_list, magnitude_list = [list(x) for x in zip(*combined_list)]
            
            # model inference
            predicted_suggestion, predicted_adjustment, predicted_magnitude = self.model(self.convert_image_list_to_tensor(image_list))

            suggested_adjustment_index = [i for i in range(len(suggestion_list)) if suggestion_list[i] == 1]
            suggestion_list = torch.tensor(suggestion_list, dtype=torch.float32).unsqueeze(1).to(self.device)            
            suggestion_loss = self.suggestion_loss_fn(suggestion_list, predicted_suggestion)

            # there are no suggestion cases
            if len(suggested_adjustment_index) == 0:
                self.optimizer.zero_grad()
                suggestion_loss.backward()
                self.optimizer.step()
                continue

            self.suggested_case_iter += 1

            suggested_adjustment_index = torch.tensor(suggested_adjustment_index).to(self.device)
            predicted_adjustment = torch.index_select(predicted_adjustment, 0, suggested_adjustment_index)
            predicted_magnitude = torch.index_select(predicted_magnitude, 0, suggested_adjustment_index)

            adjustment_list = torch.tensor(adjustment_list, dtype=torch.float32).to(self.device)
            magnitude_list = torch.tensor(magnitude_list, dtype=torch.float32).to(self.device)

            # remove no-suggestion cases
            adjustment_list = torch.index_select(adjustment_list, 0, suggested_adjustment_index)
            magnitude_list = torch.index_select(magnitude_list, 0, suggested_adjustment_index)

            adjustment_loss = self.adjustment_loss_fn(adjustment_list, predicted_adjustment)
            magnitude_loss = self.magnitude_loss_fn(magnitude_list, predicted_magnitude)

            total_loss = suggestion_loss + adjustment_loss + magnitude_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        print('\n======train end======\n')

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            tensor.append(self.transformer(image).to(self.device))
        tensor = torch.stack(tensor, dim=0)
        return tensor

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')

            epoch_log = 'epoch: %d / %d, lr: %8f' % (self.epoch, self.max_epoch, self.optimizer.param_groups[0]['lr'])
            print(epoch_log)
            
            accuracy_log = f'{self.suggestion_loss_sum/ self.train_iter:.5f}/{self.adjustment_loss_sum / self.suggested_case_iter:.5f}/{self.magnitude_loss_sum / self.suggested_case_iter:.5f}'
            print(accuracy_log)
            with open('epoch_log.txt', 'a') as f:
                f.write(accuracy_log + f"/{self.optimizer.param_groups[0]['lr']}\n")

            self.train_iter = 0
            self.suggested_case_iter = 0
            self.suggestion_loss_sum = 0
            self.adjustment_loss_sum = 0
            self.magnitude_loss_sum = 0

    def get_perturbated_image(self, data):
        image_name = data[0]
        image = Image.open(os.path.join(self.image_dir, image_name))
        best_crop_bounding_box = data[1]

        func_choice = random.randint(0, 2)
        if func_choice == 0:
            output = get_shifted_image(image, best_crop_bounding_box, allow_zero_pixel=False, option='vapnet', direction=0)
        elif func_choice == 1:
            output = get_shifted_image(image, best_crop_bounding_box, allow_zero_pixel=False, option='vapnet', direction=1)
        elif func_choice == 2 :
            output = get_rotated_image(image, best_crop_bounding_box, allow_zero_pixel=False, option='vapnet')

        if output == None:
            return None
        perturbated_image, operator = output
    
        suggestion_label = 1
        adjustment_index = -1
        adjustment_label = [0] * self.adjustment_count
        magnitude_label = [0] * self.adjustment_count

        if func_choice == 0:
            adjustment_index = 0 if operator[0] < 0 else 1
            adjustment_label[adjustment_index] = 1
            magnitude_label[adjustment_index] = -operator[0]
        if func_choice == 1:
            adjustment_index = 2 if operator[1] < 0 else 3
            adjustment_label[adjustment_index] = 1
            magnitude_label[adjustment_index] = -operator[1]
        if func_choice == 2:
            adjustment_index = 4 if operator[3] < 0 else 5
            adjustment_label[adjustment_index] = 1
            magnitude_label[adjustment_index] = -operator[3]

        return perturbated_image, suggestion_label, adjustment_label, magnitude_label

    def get_labeled_data_list(self, bc_data_list):
        image_list = []
        suggestion_label_list = []
        adjustment_label_list = []
        magnitude_label_list = []
        for data in bc_data_list:
            labeled_data = self.get_perturbated_image(data)
            if labeled_data == None:
                continue
            image_list.append(labeled_data[0])
            suggestion_label_list.append(labeled_data[1])
            adjustment_label_list.append(labeled_data[2])
            magnitude_label_list.append(labeled_data[3])

        return image_list, suggestion_label_list, adjustment_label_list, magnitude_label_list
    
    def get_best_crop_labeled_data_list(self, bc_data_list):
        image_list = []
        suggestion_label_list = []
        adjustment_label_list = []
        magnitude_label_list = []

        adjustment_label = [0] * self.adjustment_count
        magnitude_label = [0] * self.adjustment_count
        for data in bc_data_list:
            image_name = data[0]
            best_crop_bounding_box = data[1]
            image = Image.open(os.path.join(self.image_dir, image_name))
            best_crop = image.crop(best_crop_bounding_box)
            image_list.append(best_crop)
            suggestion_label_list.append(0)
            adjustment_label_list.append(adjustment_label)
            magnitude_label_list.append(magnitude_label)

        return image_list, suggestion_label_list, adjustment_label_list, magnitude_label_list

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()