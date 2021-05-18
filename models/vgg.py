from torchvision import models
from typing import Union, List, Dict, Any, cast

from absl import flags
FLAGS = flags.FLAGS

cfgs: Dict[str, List[Union[str, int]]] = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(models.VGG):
    def __init__(self, **kwargs):
        if FLAGS.depth in cfgs:
            cfg = cfgs[FLAGS.depth]
        else:
            raise RuntimeError(f"Depth {FLAGS.depth} is not supported for VGG. Select one of {', '.join(map(str,cfgs.keys()))}.")

        super(VGG, self).__init__(models.vgg.make_layers(cfg, batch_norm=FLAGS.BN), **kwargs)
