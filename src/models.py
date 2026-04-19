import segmentation_models_pytorch as smp
import torch.nn as nn

class LandVisorModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(LandVisorModel, self).__init__()
        # If we are training, use imagenet. If we are evaluating, use None.
        weights = config['model']['encoder_weights'] if pretrained else None
        
        # Check for CNN-based U-Net
        if config['model']['architecture'].lower() == "unet":
            self.model = smp.Unet(
                encoder_name=config['model']['backbone'],
                encoder_weights=weights,
                #in_channels=config['model']['in_channels'],
                classes=config['data']['num_classes']
            )
        # Check for Transformer-based SegFormer
        elif config['model']['architecture'].lower() == "segformer":
            self.model = smp.Segformer(
                encoder_name=config['model']['backbone'],
                encoder_weights=weights,
                #in_channels=config['model']['in_channels'],
                classes=config['data']['num_classes']
            )
        else:
            raise ValueError(f"Architecture {config['model']['architecture']} not supported yet!")

    def forward(self, x):
        return self.model(x)
        