import os
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from config import Config
from csnet import CSNet
from dataset_regression import SCDataset

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    sc_dataset = SCDataset('test', cfg)
    sc_loader = DataLoader(dataset=sc_dataset,
                              batch_size=8,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=False,
                              num_workers=cfg.num_workers)
    return sc_loader

class Tester(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir

        self.sc_loader = build_dataloader(self.cfg)
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

        self.sc_batch_size = self.cfg.scored_crops_batch_size

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.loss_sum = 0
        self.test_iter = 0
        self.data_length = len(self.sc_loader)

    def run(self):
        print('\n======test start======\n')
        self.model.eval().to(self.device)
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.sc_loader), total=self.data_length):
                sc_data_list = data
                sc_images = [self.make_image_crops(x[0], x[2]) for x in sc_data_list]
                sc_scores = [x[1] for x in sc_data_list]
                
                sc_images = self.convert_image_list_to_tensor(list(chain(*sc_images)))
                sc_scores = torch.tensor(list(chain(*sc_scores)))

                sc_loss = self.calculate_mse_loss(sc_images, sc_scores)
                
                self.loss_sum += sc_loss.item() if sc_loss != None else 0
                self.test_iter += 1

        print('\n======test end======\n')

        ave_loss = self.loss_sum / self.test_iter
        test_log = f'Loss: {ave_loss:.5f}'
        # wandb.log({"test_loss": ave_loss})
        with open('./regression_test_log.txt', 'a') as f:
            f.write(f'{ave_loss}\n')
        print(test_log)

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            tensor.append(self.transformer(image))
        tensor = torch.stack(tensor, dim=0)
        return tensor

    def calculate_mse_loss(self, images, scores):
        images = images.to(self.device)
        scores = scores.to(self.device)

        scores = scores.view(scores.shape[0], 1)
        predicted_scores = self.model(images)
        loss = self.loss_fn(predicted_scores, scores)

        return loss
            
    def make_image_crops(self, image, crops_list):
        image_crops_list = []
        for crop in crops_list:
            image_crops_list.append(image.crop(crop))
        return image_crops_list
    
def test_while_training():
    cfg = Config()

    model = CSNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'regression_checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()

if __name__ == '__main__':
    cfg = Config()
    """
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.learning_rate,
        "architecture": "CNN",
        "dataset": "SC_dataset",
        "epochs": cfg.max_epoch,
        "memo": "failed"
        }
    )
    """
    model = CSNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'regression_checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()

    wandb.finish()