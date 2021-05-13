import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_float  (name = "weightDecay" , default = 0.0005       , help = "L2 weight decay.")
flags.DEFINE_float  (name = "momentum"    , default = 0.9          , help = "SGD Momentum.")

class SGD(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(lr           = FLAGS.learningRate,
                        momentum     = FLAGS.momentum,
                        weight_decay = FLAGS.weightDecay,
                        **kwargs)
        super(SGD, self).__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)

    def step(self, zeroGrad: bool = True):
        self.base_optimizer.step()
        if zeroGrad: self.zero_grad()

    def load_state_dict(self, state_dict):
        super(SGD, self).load_state_dict(state_dict)
        self.base_optimizer.__setstate__(self.__getstate__())

    def state_dict(self):
        return self.base_optimizer.state_dict()
