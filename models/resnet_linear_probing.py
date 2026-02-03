import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class ResNetLP(nn.Module):

    def __init__(self, base_model, out_dim,checkpoint_path, freeze_backbone=True):
        super(ResNetLP, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        self.backbone = self._get_basemodel(base_model)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # Remove 'backbone.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.') and not k.startswith('backbone.fc'):
                new_key = k[len('backbone.'):]
                new_state_dict[new_key] = v
                
        self.backbone.load_state_dict(new_state_dict, strict=False)
        
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(dim_mlp, out_dim)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
