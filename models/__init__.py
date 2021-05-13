from .wideResNet import WideResNet
from .vanillaNet import VanillaNet
from .vgg import VGG

# Dict could be used to select model without switch/case
modelDict = {
    "WRN"    : WideResNet,
    "vanilla": VanillaNet,
    "VGG"    : VGG,
}

# Define flags here that are used by multiple models to avoid double definitions.
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_enum   (name = "model"        , default = "WRN"   , enum_values = modelDict.keys(), help="model name (WRN = WideResNet)")
flags.DEFINE_float  (name = "dropout"      , default = 0.0          , help = "Dropout rate.")
flags.DEFINE_bool   (name = "BN"           , default = False          , help = "use batch norm ?")
flags.DEFINE_integer(name = "depth"       , default = 16           , help = "Number of layers.")

