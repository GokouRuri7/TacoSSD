import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

def vgg(pretrained = False):
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 19, 19, 512 -> 19, 19, 512 
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 19, 19, 512 -> 19, 19, 1024
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # 19, 19, 1024 -> 19, 19, 1024
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    model = nn.ModuleList(layers)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        state_dict = {k.replace('features.', '') : v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict = False)
    return model

if __name__ == "__main__":
    net = vgg()
    for i, layer in enumerate(net):
        print(i, layer)