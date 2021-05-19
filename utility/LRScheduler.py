import math
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_float  (name = "learningRate", default = 0.1               , help = "Base learning rate at the start of the training.")
flags.DEFINE_list   (name = "lrScheduler"   , default = ["CWR"]  , help = "list of learning rate schedulers")


class _LRScheduler():
    def __init__(self, optimizer, last_epoch = -1):
        self.optimizer = optimizer
        self.base_lrs = [FLAGS.learningRate for _ in optimizer.param_groups]
        self.last_epoch = last_epoch

    def _calcFactor(self, progress):
        # self.factor = self.last_epoch * ...
        raise NotImplementedError
        
    def step(self, epoch = None, progress = 0):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self._calcFactor(progress)

        for group, baseLR in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = baseLR * self.factor

    def get_last_lr(self):
        return [baseLR * self.factor for baseLR in self.base_lrs]

class ConstLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(ConstLR, self).__init__(optimizer, last_epoch = last_epoch)

    def _calcFactor(self, progress):
        self.factor = 1


flags.DEFINE_integer(name = "LRScheduler_step_steps"   , default = 3              , help = "")
flags.DEFINE_float(name = "LRScheduler_step_gamma"   , default = 0.2              , help = "")
class StepLR(_LRScheduler):
    def __init__(self, optimizer, steps = 3, last_epoch = -1):
        super(StepLR, self).__init__(optimizer, last_epoch = last_epoch)
        self.step_size = FLAGS.epochs//(FLAGS.LRScheduler_step_steps+1)
        self.gamma = FLAGS.LRScheduler_step_gamma

    def _calcFactor(self, progress):
        self.factor = self.gamma ** (self.last_epoch // self.step_size)

flags.DEFINE_float  (name = "LRScheduler_exp_maxDecay"   , default = 0.01              , help = "Decay that is to be reached at the end of training.")
class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(ExponentialLR, self).__init__(optimizer, last_epoch = last_epoch)
    
        self.gamma = FLAGS.LRScheduler_exp_maxDecay**(1/FLAGS.epochs)

    def _calcFactor(self, progress):
        self.factor = self.gamma ** (self.last_epoch + progress)




flags.DEFINE_float  (name = "LRScheduler_cos_periods"   , default = 0.5              , help = "Periods of cosine.")
class CosineLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(CosineLR, self).__init__(optimizer, last_epoch = last_epoch)

        self.epochsPerPeriod = FLAGS.epochs / FLAGS.LRScheduler_cos_periods

    def _calcFactor(self, progress):
        self.factor = 0.5 * (1 + math.cos((self.last_epoch + progress) / self.epochsPerPeriod * 2*math.pi))


flags.DEFINE_float  (name = "LRScheduler_WRN_T0"   , default = 50              , help = "Number of epochs for the first restart.")
flags.DEFINE_float  (name = "LRScheduler_WRN_Tmult", default = 1.0               , help = "Factor by which the period length is increased after each restart.")
class CosineWarmRestartsLR(_LRScheduler):
    """
        Args:
        T_0 (float): Number of epochs for the first restart.
        T_mult (int, optional): A factor increases after a restart. Default: 1.
    """
    def __init__(self, optimizer, last_epoch = -1):
        super(CosineWarmRestartsLR, self).__init__(optimizer, last_epoch = last_epoch)

        self.T0    = FLAGS.LRScheduler_WRN_T0
        self.Tmult = FLAGS.LRScheduler_WRN_Tmult


        # closed form of
        # self.factor = 0.5 * (1 + math.cos(d/Ti *math.pi))
        # where d ist distance to last restart, Ti is current period length
        # and restart epochs are given by E = T_0 * sum_x=0^N T_m^x ; x \in N
        if FLAGS.LRScheduler_WRN_Tmult == 1:
            def __calcFactor(progress: float):
                x = int(((self.last_epoch + progress) - self.T0 )/self.T0  + 1)
                d = (self.last_epoch + progress) - x * self.T0 
                Ti = self.T0 
                self.factor = 0.5 * (1 + math.cos(d/Ti *math.pi))
        else:
            def __calcFactor(progress: float):
                x = int(np.log(1-(1-self.Tmult)*(self.last_epoch + progress)/self.T0)/np.log(self.Tmult)) - 1
                d = (self.last_epoch + progress) - (1-self.Tmult**(x+1))/(1-self.Tmult) * self.T0
                Ti = self.T0 * self.Tmult**(x+1)
                self.factor = 0.5 * (1 + math.cos(d/Ti *math.pi))

        self._calcFactor = __calcFactor

class ChainedScheduler(_LRScheduler):
    def __init__(self, optimizer, schedulerClasses: list, last_epoch = -1):
        super(ChainedScheduler, self).__init__(optimizer, last_epoch = last_epoch)

        self.schedulers = [scheduler(optimizer, last_epoch = last_epoch) for scheduler in schedulerClasses]
        
    def _calcFactor(self, progress):
        self.factor = 1
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.last_epoch
            scheduler._calcFactor(progress)
            self.factor *= scheduler.factor


schedulerDict = {
    "const": ConstLR,
    "step" : StepLR,
    "exp"  : ExponentialLR,
    "cos"  : CosineLR,
    "CWR"  : CosineWarmRestartsLR,
}


def getLRScheduler(optimizer):
    schedulers = []
    for scheduler in  FLAGS.lrScheduler:
        if scheduler in schedulerDict:
            schedulers.append(schedulerDict[scheduler])
        else:
            raise RuntimeError(f"LR Scheduler {scheduler} not found. Available schedulers: {', '.join(schedulerDict.keys())}")

    if len(schedulers) == 0:
        raise RuntimeError(f"No LR Scheduler selected")
    elif len(schedulers) == 1:
        return schedulers[0](optimizer)
    else:
        return ChainedScheduler(optimizer, schedulers)


if __name__ == '__main__':
    import sys
    import torch
    import matplotlib.pylab as plt
    import numpy as np
    
    flags.DEFINE_integer(name = "epochs"      , default = 160          , help = "Total number of epochs.")
    FLAGS(sys.argv)
    optimizer = torch.optim.SGD(torch.nn.Linear(5,5).parameters(), lr = FLAGS.learningRate)
    lrScheduler = getLRScheduler(optimizer)

    batches = 10
    lr = []
    epochs = []
    for epoch in range(1, FLAGS.epochs + 1):
        for batch in range(batches):
            lr.append(optimizer.param_groups[0]["lr"])
            lrScheduler.step(epoch-1, (batch+1)/batches)
            epochs.append(epoch + batch/batches)

    fig, ax = plt.subplots()

    ax.plot(epochs, lr, "r-")

    plt.show()
