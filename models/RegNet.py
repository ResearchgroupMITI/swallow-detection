import torchvision.models as models
import torch.nn as nn
# import torch
#from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
# from datetime import datetime

class RegNetModel(nn.Module):
    def __init__(self, hparams):
        super(RegNetModel, self).__init__()
        if hparams.pretrained:
            weights = models.RegNet_Y_400MF_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.model = models.regnet_y_400mf(weights=weights)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=hparams.dropout_rate),
            nn.Linear(440, hparams.num_classes)
        )

    def forward(self, x):
        out_stem = self.model(x)

        return out_stem

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet18model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet18model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet18model_specific_args.add_argument("--dropout_rate",
                                                 type=float,
                                                 default=0.2,
                                                 help="dropout rate")
        return parser
