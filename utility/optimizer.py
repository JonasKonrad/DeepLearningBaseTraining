import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_float  (name = "weightDecay" , default = 0.00001       , help = "L2 weight decay.")
flags.DEFINE_float  (name = "momentum"    , default = 0.9          , help = "SGD Momentum.")
flags.DEFINE_enum   (name = "optimzer"   , default = "SGD"   , enum_values = ["SGD"], help="Dataset")
flags.DEFINE_integer(name = "batchSizeMult", default = 1 , help = "Effective Batch Size is multiplied by this factor by applying gradient step only after 'batchSizeMult' iterations.")

class SGD(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(lr           = FLAGS.learningRate,
                        momentum     = FLAGS.momentum,
                        weight_decay = FLAGS.weightDecay,
                        **kwargs)
        super(SGD, self).__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)

        self.batchSizeMult = FLAGS.batchSizeMult
        self.batchCounter = 0

    def step(self):
        if self.batchSizeMult == 1:
            #handle == 1 case seperately for better performance 
            self.base_optimizer.step()
            self.zero_grad()
        else:
            self.batchCounter += 1
            if self.batchCounter % self.batchSizeMult == 0:
                self.batchCounter = 0

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.grad.div_(self.batchSizeMult)

                self.base_optimizer.step()
                self.zero_grad()

    def load_state_dict(self, state_dict):
        super(SGD, self).load_state_dict(state_dict)
        self.base_optimizer.__setstate__(self.__getstate__())

    def state_dict(self):
        return self.base_optimizer.state_dict()

