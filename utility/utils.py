import random
import torch


from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool   (name = "rndSeed"       , default = False        , help = "Whether to set rnd seed.")
flags.DEFINE_bool   (name = "deterministic" , default = False        , help = "Whether use deterministic algorithms. May decrease performance.")


def initialize():
    if FLAGS.rndSeed:
        seed = random.random()
    else:
        seed = 42

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True

    if FLAGS.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False