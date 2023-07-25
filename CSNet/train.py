from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import os

from config import Config
from csnet import CSNet
from dataset import *
from augmentation import *
from image_preprocess import get_cropping_image, get_zooming_out_image, get_shifted_image, get_rotated_image

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

        self.sc_loader, self.bc_loader, self.un_loader = build_dataloader(self.cfg)
        # self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.sc_batch_size = self.cfg.scored_crops_batch_size
        self.bc_batch_size = self.cfg.best_crop_K
        self.un_batch_size = self.cfg.unlabeled_P
        
        self.iter = 0

        self.loss_fn = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
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


    def training(self):
        # self.model.train().to(self.device)
        self.model.train()
        print('\n======train start======\n')

        for index, data in enumerate(zip(self.sc_loader, self.bc_loader, self.un_loader)):
            sc_data_list = data[0]
            bc_data_list = data[1]
            un_data_list = data[2]

            sc_pos_images, sc_neg_images = self.make_pairs_scored_crops(sc_data_list[0])
            sc_loss = self.calculate_pairwise_ranking_loss(sc_pos_images, sc_neg_images)
            
            bc_pos_images, bc_neg_images = self.make_pairs_perturbating(bc_data_list, labeled=True)
            bc_loss = self.calculate_pairwise_ranking_loss(bc_pos_images, bc_neg_images)

            un_pos_images, un_neg_images = self.make_pairs_perturbating(un_data_list, labeled=False)
            un_loss = self.calculate_pairwise_ranking_loss(un_pos_images, un_neg_images)

            total_loss = sc_loss + bc_loss + un_loss

            """
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            """

    def calculate_pairwise_ranking_loss(self, pos_images, neg_images):
        target = torch.ones((pos_images.shape[0], 1))
        print(self.model(pos_images))
        print(self.model(neg_images))
        loss = self.loss_fn(self.model(pos_images), self.model(neg_images), target=target)
        return loss

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')
            

    def make_pairs_scored_crops(self, data):
        image_name = data[0]
        crops_list = data[1]

        image = Image.open(os.path.join(self.image_dir, image_name))

        # sort in descending order by score
        sorted_crops_list = sorted(crops_list, key = lambda x: -x['score'])
        
        boudning_box_pairs = [(sorted_crops_list[i]['crop'], sorted_crops_list[j]['crop']) for i in range(len(sorted_crops_list)) for j in range(i + 1, len(sorted_crops_list))]
        
        pos_images = []
        neg_images = []
        for pos_box, neg_box in boudning_box_pairs:
            pos_image = image.crop(pos_box)
            neg_image = image.crop(neg_box)
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled=True)

            pos_images.append(self.transformer(pos_image))
            pos_images.append(self.transformer(augmented_pos_image))
            neg_images.append(self.transformer(neg_image))
            neg_images.append(self.transformer(augmented_neg_image))

        pos_images = torch.stack(pos_images, dim=0)
        neg_images = torch.stack(neg_images, dim=0)
        return pos_images, neg_images

    def make_pair_perturbating(self, data, labeled=True):
        if labeled == True:
            image_name = data[0]
            image = Image.open(os.path.join(self.image_dir, image_name))
            best_crop_bounding_box = data[1]
            best_crop = image.crop(best_crop_bounding_box)
        else:
            image_name = data
            image = Image.open(os.path.join(self.image_dir, image_name))
            best_crop_bounding_box = [0, 0, image.size[0], image.size[1]]
            best_crop = image

        # bb가 끝점에 걸칠 때 예외처리 필요함
        func_list = [get_rotated_image, get_shifted_image, get_zooming_out_image, get_cropping_image]
        perturbate_func = random.choice(func_list)

        allow_zero_pixel = not labeled

        perturbated_image = perturbate_func(image, best_crop_bounding_box, allow_zero_pixel, option='csnet')

        return best_crop, perturbated_image

    def make_pairs_perturbating(self, data_list, labeled):
        print(data_list)
        pos_images = []
        neg_images = []
        for data in data_list:
            pos_image, neg_image = self.make_pair_perturbating(data, labeled)
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled)

            pos_images.append(self.transformer(pos_image))
            neg_images.append(self.transformer(neg_image))

            pos_images.append(self.transformer(augmented_pos_image))
            neg_images.append(self.transformer(augmented_neg_image))

        pos_images = torch.stack(pos_images, dim=0)
        neg_images = torch.stack(neg_images, dim=0)

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
    # weight_file = os.path.join(cfg.weight_dir, 'csnet_checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.training()
    # trainer.run()