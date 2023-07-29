import torch.nn as nn
import torch
import torchvision.models
import math
import os
from torchvision.transforms import transforms

from CSNet.config import Config

class CSNet(nn.Module):
    def __init__(self, cfg):
        super(CSNet, self).__init__()
        self.cfg = cfg

        self.backbone = self.build_backbone(pretrained=True)

        self.spp_pool_size = [5, 2, 1]
        
        self.last_layer = nn.Sequential(
            nn.Linear(38400, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        

    def forward(self, image):
        feature_map = self.backbone(image)
        spp = self.spatial_pyramid_pool(feature_map, feature_map.shape[0], [int(feature_map.size(2)),int(feature_map.size(3))],self.spp_pool_size)
        feature_vector = self.last_layer(spp)

        output = self.output_layer(feature_vector)
        return output
    
    def build_backbone(self, pretrained):
        model = torchvision.models.mobilenet_v2(pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
        return backbone
    
    # parameter: tensor, batch_size, tensor width and height, spp pool size
    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view([num_sample, -1])
            else:
                spp = torch.cat((spp, x.view([num_sample, -1])), 1)
        return spp
    
def get_pretrained_CSNet():
    cfg = Config()

    model = CSNet(cfg)
    model.eval()
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    return model
    
if __name__ == '__main__':
    cfg = Config()
    model = CSNet(cfg)
    x = torch.randn((1, 3, 299, 299))
    output = model(x)
    torch.save(model.state_dict(), './output/weight/csnet_checkpoint.pth')
    print(output)
    print(output[0].item())
