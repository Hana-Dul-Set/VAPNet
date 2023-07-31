from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import os
import time
from torchvision.transforms import transforms

from config import Config
from csnet import CSNet
from dataset import SCDataset, BCDataset, UNDataset
from image_utils.augmentation import *
from image_utils.image_preprocess import get_cropping_image, get_zooming_image, get_shifted_image, get_rotated_image
from test import test_while_training

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    sc_dataset = SCDataset('train', cfg)
    sc_loader = DataLoader(dataset=sc_dataset,
                              batch_size=cfg.scored_crops_batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    bc_dataset = BCDataset('train', cfg)
    bc_loader = DataLoader(dataset=bc_dataset,
                              batch_size=cfg.best_crop_K,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    un_dataset = UNDataset('train', cfg)
    un_loader = DataLoader(dataset=un_dataset,
                              batch_size=cfg.unlabeled_P,
                              shuffle=True,
                              collate_fn=not_convert_to_tesnor,
                              num_workers=cfg.num_workers)
    return sc_loader, bc_loader, un_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        # self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.sc_batch_size = self.cfg.scored_crops_batch_size
        self.bc_batch_size = self.cfg.best_crop_K
        self.un_batch_size = self.cfg.unlabeled_P
        
        self.iter = 0

        self.loss_fn = torch.nn.MarginRankingLoss(margin=self.cfg.pairwise_margin, reduction='mean')

        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)
        """
        self.optimizer = optim.SGD(params=model.parameters(),
                                   lr=self.cfg.learning_rate,
                                   weight_decay=self.cfg.weight_decay)
        """
        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch

        self.total_loss_sum = 0
        self.sc_loss_sum = 0
        self.bc_loss_sum = 0
        self.un_loss_sum = 0
        self.train_iter = 0
        self.sc_iter = 0
        self.bc_iter = 0
        self.un_iter = 0

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])


    def training(self):
        self.model.train().to(self.device)
        # self.model.train()
        self.sc_loader, self.bc_loader, self.un_loader = build_dataloader(self.cfg)
        print('\n======train start======\n')

        for index, data in enumerate(zip(self.sc_loader, self.bc_loader, self.un_loader)):
            sc_data_list = data[0]
            bc_data_list = data[1]
            un_data_list = data[2]

            sc_pos_images, sc_neg_images = self.make_pairs_scored_crops(sc_data_list[0])
            if len(sc_pos_images) == 0:
                sc_loss = None
            else:
                sc_loss = self.calculate_pairwise_ranking_loss(sc_pos_images, sc_neg_images)
            
            bc_pos_images, bc_neg_images = self.make_pairs_perturbating(bc_data_list, labeled=True)
            if len(bc_pos_images) == 0:
                bc_loss = None
            else:
                bc_loss = self.calculate_pairwise_ranking_loss(bc_pos_images, bc_neg_images)
            
            un_pos_images, un_neg_images = self.make_pairs_perturbating(un_data_list, labeled=False)
            un_loss = self.calculate_pairwise_ranking_loss(un_pos_images, un_neg_images)

            total_loss = 0
            compare_loss = 0
            if sc_loss != None:
                total_loss += sc_loss
                self.sc_iter += 1
                compare_loss = sc_loss.item()
            if False:
                total_loss += bc_loss
                self.bc_iter += 1
                compare_loss = bc_loss.item()
            """
            total_loss += un_loss
            self.un_iter += 1
            """

            """
            if sc_loss == None and bc_loss == None:
                total_loss = un_loss
            elif bc_loss == None:
                total_loss = sc_loss + un_loss
                self.sc_iter += 1
            elif sc_loss == None:
                total_loss = bc_loss + un_loss
                self.bc_iter += 1
            else:
                total_loss = sc_loss + bc_loss + un_loss
                self.sc_iter += 1
                self.bc_iter += 1
            """

            loss_log = f'L_SC: {sc_loss.item() if sc_loss != None else 0.0:.5f}, L_BC: {bc_loss.item() if bc_loss != None else 0.0:.5f}, L_UN: {un_loss.item():.5f}, Total Loss: {total_loss.item():.5f}'
            self.total_loss_sum += total_loss.item() 
            self.sc_loss_sum += sc_loss.item() if sc_loss != None else 0
            self.bc_loss_sum += bc_loss.item() if bc_loss != None else 0
            self.un_loss_sum += un_loss.item()
            self.train_iter += 1
            print(loss_log)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            del sc_loss
            del bc_loss
            del un_loss
            del total_loss
            sc_loss = None
            bc_loss = None
            un_loss = None
            total_loss = None

        print('\n======train end======\n')

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            tensor.append(self.transformer(image))
        tensor = torch.stack(tensor, dim=0)
        
        return tensor

    def calculate_pairwise_ranking_loss(self, pos_images, neg_images):
        pos_tensor = self.convert_image_list_to_tensor(pos_images)
        neg_tensor = self.convert_image_list_to_tensor(neg_images)
        tensor_concat = torch.cat((pos_tensor, neg_tensor), dim=0)
        tensor_concat = tensor_concat.to(self.device)
        
        """
        pos_tensor = self.model(pos_tensor.to(self.device))
        neg_tensor = self.model(neg_tensor.to(self.device))
        """
        tensor_concat = self.model(tensor_concat)
        pos_tensor, neg_tensor = torch.split(tensor_concat, [tensor_concat.shape[0] // 2, tensor_concat.shape[0] // 2])
        target = torch.ones((pos_tensor.shape[0], 1)).to(self.device)
        loss = self.loss_fn(pos_tensor, neg_tensor, target=target)
        del pos_tensor
        del neg_tensor
        del target
        return loss

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
            
            accuracy_log = f'{self.sc_loss_sum / self.sc_iter:.5f}/{self.bc_loss_sum / self.bc_iter:.5f}/{self.un_loss_sum / self.train_iter:.5f}/{self.total_loss_sum / self.train_iter:.5f}'
            print(accuracy_log)
            with open('epoch_log.txt', 'a') as f:
                f.write(accuracy_log + f"/{self.optimizer.param_groups[0]['lr']}\n")

            self.train_iter = 0
            self.total_loss_sum = 0
            self.sc_loss_sum = 0
            self.bc_loss_sum = 0
            self.un_loss_sum = 0
            self.sc_iter = 0
            self.bc_iter = 0
            if self.epoch % 1 == 0:
                test_while_training()

    def shuffle_two_lists_in_same_order(self, list1, list2):
        combined_lists = list(zip(list1, list2))
        random.shuffle(combined_lists)
        shuffled_list1, shuffled_list2 = zip(*combined_lists)

        return list(shuffled_list1), list(shuffled_list2)
            
    def make_pairs_scored_crops(self, data):
        image_name = data[0]
        crops_list = data[1]

        image = Image.open(os.path.join(self.image_dir, image_name))

        # sort in descending order by score
        sorted_crops_list = sorted(crops_list, key = lambda x: -x['score'])
        
        boudning_box_pairs = []
        for i in range(len(sorted_crops_list)):
            for j in range(i + 1, len(sorted_crops_list)):
                if sorted_crops_list[i]['score'] == sorted_crops_list[j]['score']:
                    continue
                boudning_box_pairs.append((sorted_crops_list[i]['crop'], sorted_crops_list[j]['crop']))

        pos_images = []
        neg_images = []
        for pos_box, neg_box in boudning_box_pairs:
            pos_image = image.crop(pos_box)
            neg_image = image.crop(neg_box)
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled=True)

            pos_images.append(pos_image)
            neg_images.append(neg_image)

            # pos_images.append(augmented_pos_image)
            # neg_images.append(augmented_neg_image)

        if len(pos_images) != 0:
            pos_images, neg_images = self.shuffle_two_lists_in_same_order(pos_images, neg_images)

        return pos_images, neg_images

    def make_pair_perturbating(self, data, labeled=True):
        if labeled == True:
            image_name = data[0]
            image = Image.open(os.path.join(self.image_dir, image_name))
            best_crop_bounding_box = data[1]
            best_crop = image.crop(best_crop_bounding_box)
        else:
            image_name = data
            image = Image.open(os.path.join(os.path.join(self.image_dir, 'unlabeled'), image_name))
            best_crop_bounding_box = [0, 0, image.size[0], image.size[1]]
            best_crop = image

        func_list = [get_rotated_image, get_shifted_image, get_zooming_image, get_cropping_image]
        perturbate_func = random.choice(func_list)

        allow_zero_pixel = not labeled

        perturbated_image = perturbate_func(image, best_crop_bounding_box, allow_zero_pixel, option='csnet')
        if perturbated_image == None:
            return None

        return best_crop, perturbated_image

    def make_pairs_perturbating(self, data_list, labeled):
        pos_images = []
        neg_images = []
        for data in data_list:
            image_pair = self.make_pair_perturbating(data, labeled)
            if image_pair == None:
                continue

            pos_image = image_pair[0]
            neg_image = image_pair[1]
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled)

            pos_images.append(pos_image)
            neg_images.append(neg_image)

            pos_images.append(augmented_pos_image)
            neg_images.append(augmented_neg_image)

        if len(pos_images) != 0:
            pos_images, neg_images = self.shuffle_two_lists_in_same_order(pos_images, neg_images)

        return pos_images, neg_images

    def augment_pair(self, image_pair, labeled=True):
        pos_image = image_pair[0]
        neg_image = image_pair[1]
        func_list = [shift_borders, zoom_out_borders, rotation_borders]
        augment_func = random.choice(func_list)
        if labeled:
            augment_pos_image = augment_func(pos_image)
            augment_neg_image = augment_func(neg_image)
        else:
            augment_pos_image = augment_func(pos_image)
            augment_neg_image = neg_image

        return augment_pos_image, augment_neg_image

if __name__ == '__main__':
    cfg = Config()

    model = CSNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()