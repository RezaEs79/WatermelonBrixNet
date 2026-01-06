from .fusion_net import FusionNet
from .gated_net import GatedNet
from .swin_net import SwinNet
from .cross_net import CrossNet

def build_model(model_name, num_classes=1):
    """
    Factory function to initialize models.
    """
    name = model_name.lower()
    
    if name == 'fusion':
        return FusionNet(num_classes=num_classes)
    elif name == 'gated':
        return GatedNet(num_classes=num_classes)
    elif name == 'swin':
        return SwinNet(num_classes=num_classes)
    elif name == 'cross': 
        return CrossNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Available: fusion, gated, swin, cross")