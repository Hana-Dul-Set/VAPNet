import torch.nn as nn
import torch
import torchvision.models
from config import Config
import math

class VAPNet(nn.Module):
    def __init__(self, cfg):
        super(VAPNet, self).__init__()
        self.cfg = cfg

        self.backbone = self.build_backbone(pretrained=True)

        self.spp_pool_size = [5, 2, 1]
        self.adjustment_count = 6
        
        self.last_layer = nn.Sequential(
            nn.Linear(38400, 1024),
            nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )

        self.suggestion_output_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.adjustment_output_layer = nn.Sequential(
            nn.Linear(1024, self.adjustment_count),
            nn.Softmax(dim=1)
        )

        self.magnitude_output_layer = nn.Sequential(
            nn.Linear(1024, self.adjustment_count),
        )
        
    def forward(self, image):
        feature_map = self.backbone(image)
        spp = self.spatial_pyramid_pool(feature_map, feature_map.shape[0], [int(feature_map.size(2)),int(feature_map.size(3))],self.spp_pool_size)
        feature_vector = self.last_layer(spp)

        suggestion_predictor = self.suggestion_output_layer(feature_vector)
        adjustment_predictor = self.adjustment_output_layer(feature_vector)
        magnitude_predictor = self.magnitude_output_layer(feature_vector)

        return suggestion_predictor, adjustment_predictor, magnitude_predictor
    
    def build_backbone(self, pretrained):
        model = torchvision.models.mobilenet_v2(pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
        return backbone
    
    # parameter: tensor, batch_size, tensor width and height, spp pool size
    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        for i in range(len(out_pool_size)):
            """
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid))
            x = maxpool(previous_conv)
            """
            maxpool = nn.AdaptiveMaxPool2d((out_pool_size[i], out_pool_size[i]))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view([num_sample, -1])
            else:
                spp = torch.cat((spp, x.view([num_sample, -1])), 1)
        return spp
    
if __name__ == '__main__':
    cfg = Config()
    model = VAPNet(cfg)
    x = torch.randn((1, 3, 224, 224))
    output = model(x)
    print(output)
    print(output[0].item())
    print(output[1][0].tolist())
    print(output[2][0].tolist())
