from .wideResNet import WideResNet
from .vanillaNet import VanillaNet

# Dict could be used to select model without switch/case
modelDict = {
    "WRN"    : WideResNet,
    "vanilla": VanillaNet,
}

# Define flags here that are used by multiple models to avoid double definitions.
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_enum   (name = "model"        , default = "WRN"   , enum_values = modelDict.keys(), help="model name (WRN = WideResNet)")
flags.DEFINE_float  (name = "dropout"      , default = 0.0          , help = "Dropout rate.")
flags.DEFINE_bool   (name = "BN"           , default = False          , help = "use batch norm ?")

