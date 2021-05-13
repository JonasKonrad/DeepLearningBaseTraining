from torchvision import models
from typing import Union, List, Dict, Any, cast

from absl import flags
FLAGS = flags.FLAGS

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 'M', 128, 'M','M','M','M'],
    'G': [64, 'M', 128, 'M', 256, 256, 'M','M','M'],
    'H': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M','M'],
}


class VGG(models.VGG):
    def __init__(self, **kwargs):
        cfg_names = {
            5: "F",
            7: "G",
            9: "H",
            11: "A",
            13: "B",
            16: "D",
            19: "E",
        }
        if FLAGS.depth in cfg_names:
            cfg = cfg_names[FLAGS.depth]
        else:
            raise RuntimeError(f"Depth {FLAGS.depth} is not supported for VGG. Select one of {', '.join(cfg_names.keys())}.")

        super(VGG, self).__init__(models.vgg.make_layers(cfgs[cfg], batch_norm=FLAGS.BN), **kwargs)
