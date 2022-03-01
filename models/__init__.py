from .wideResNet import WideResNet
from .vanillaNet import VanillaNet
from .efficientNet import EfficientNet
from .vgg import VGG


def effNetWrapper(num_classes, *args, **kwargs):
    return EfficientNet.from_name('efficientnet-b7', num_classes=num_classes)

def effNetWrapper_pretrained(num_classes, *args, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)


# Dict could be used to select model without switch/case
modelDict = {
    "WRN"              : WideResNet,
    "vanilla"          : VanillaNet,
    "VGG"              : VGG,
    "EfficientNet"     : effNetWrapper,
    "EfficientNet_pre" : effNetWrapper_pretrained,
}


from utility.args import Args
# Define flags here that are used by multiple models to avoid double definitions.
Args.add_argument("--model", type=str, help="model name (WRN = WideResNet")
Args.add_argument("--dropout", type=float, help="Dropout rate.")
Args.add_argument("--BN", type=bool, help="use batch norm ?")
Args.add_argument("--depth", type=int, help="Number of layers.")
