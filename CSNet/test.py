import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from config import Config
from csnet import CSNet
from dataset import SCDataset

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    sc_dataset = SCDataset('test', cfg)
    sc_loader = DataLoader(dataset=sc_dataset,
                              batch_size=cfg.scored_crops_batch_size,
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

        self.loss_fn = torch.nn.MarginRankingLoss(margin=self.cfg.pairwise_margin, reduction='mean')

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.loss_sum = 0
        self.correct_prediction_counts = 0
        self.total_prediction_counts = 0
        self.test_iter = 0
        self.data_length = len(self.sc_loader)


    def run(self):
        print('\n======test start======\n')
        self.model.eval().to(self.device)
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.sc_loader), total=self.data_length):
                sc_data_list = data
                sc_pos_images, sc_neg_images = self.make_pairs_scored_crops(sc_data_list[0])
                if len(sc_pos_images) == 0:
                    continue
                sc_loss, correct_prediction, total_prediction = self.calculate_loss_and_accuracy(sc_pos_images, sc_neg_images)

                self.loss_sum += sc_loss.item() 
                self.total_prediction_counts += total_prediction
                self.correct_prediction_counts += correct_prediction.item()
                self.test_iter += 1

        print('\n======test end======\n')

        ave_loss = self.loss_sum / self.test_iter
        accuracy = self.correct_prediction_counts / self.total_prediction_counts
        test_log = f'Loss: {ave_loss:.5f}, Accuracy: {accuracy *  100:.2f} %'
        wandb.log({"test_loss": ave_loss, "accuracy": accuracy})
        with open('./test_log.txt', 'a') as f:
            f.write(f'{ave_loss}/{accuracy}\n')
        print(test_log)

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            tensor.append(self.transformer(image))
        tensor = torch.stack(tensor, dim=0)
        return tensor

    def calculate_loss_and_accuracy(self, pos_images, neg_images):
        pos_tensor = self.convert_image_list_to_tensor(pos_images)
        neg_tensor = self.convert_image_list_to_tensor(neg_images)

        pos_tensor = pos_tensor.to(self.device)
        neg_tensor = neg_tensor.to(self.device)

        pos_scores = [self.model(x.unsqueeze(0)) for x in pos_tensor]
        neg_scores = [self.model(x.unsqueeze(0)) for x in neg_tensor]
        pos_scores = torch.cat(pos_scores, dim=0)
        neg_scores = torch.cat(neg_scores, dim=0)
        target = torch.ones((pos_scores.shape[0], 1)).to(self.device)

        loss = self.loss_fn(pos_scores, neg_scores, target=target)

        total_prediction_counts = pos_tensor.shape[0]
        comparison_result = pos_scores > neg_scores
        correct_prediction_counts = comparison_result.sum(dim=0)
        return loss, correct_prediction_counts, total_prediction_counts
            
    def make_pairs_scored_crops(self, data):
        image = data[0]
        crops_list = data[1]

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

            pos_images.append(pos_image)
            neg_images.append(neg_image)

        return pos_images, neg_images
    
def test_while_training():
    cfg = Config()

    model = CSNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()

if __name__ == '__main__':
    cfg = Config()

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

    model = CSNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()

    wandb.finish()