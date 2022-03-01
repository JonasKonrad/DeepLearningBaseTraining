import torch
from utility.args import Args


Args.add_argument("--nesterov", type=bool, help="Whether to use nesterov momentum in SGD.")
Args.add_argument("--batchSizeMult", type=int, help="Effective Batch Size is multiplied by this factor by applying gradient step only after 'batchSizeMult' iterations.")
Args.add_argument("--optimzer", type=str, help="")
Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
Args.add_argument("--momentum", type=float, help="SGD Momentum.")

class SGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        defaults = dict(lr           = Args.learningRate,
                        momentum     = Args.momentum,
                        weight_decay = Args.weightDecay,
                        nesterov     = Args.nesterov,
                        **kwargs)
        super(SGD, self).__init__(params, **defaults)
        self.batchSizeMult = Args.batchSizeMult
        self.batchCounter = 0

    @torch.no_grad()
    def step(self):
        if self.batchSizeMult == 1:
            #handle == 1 case seperately for better performance 
            super(SGD, self).step()
            self.zero_grad()
        else:
            self.batchCounter += 1
            if self.batchCounter % self.batchSizeMult == 0:
                self.batchCounter = 0

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.grad.div_(self.batchSizeMult)

                super(SGD, self).step()
                self.zero_grad()
