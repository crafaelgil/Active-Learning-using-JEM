from models import wideresnet
from models import resnet


def basenet(network="wideresnet", n_ch=3):
    if network == "wideresnet":
        return wideresnet.Wide_ResNet(depth=28, widen_factor=10, norm=None, dropout_rate=0.0, input_channels=n_ch)
    elif network == "resnet18":
        return resnet.ResNet18(in_channels=n_ch)
    elif network == "resnet50":
        return resnet.ResNet50(in_channels=n_ch)
