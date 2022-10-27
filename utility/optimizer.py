import torch
from utility.args import Args


Args.add_argument("--nesterov", type=bool, help="Whether to use nesterov momentum in SGD.")
Args.add_argument("--batchSizeMult", type=int, help="Effective Batch Size is multiplied by this factor by applying gradient step only after 'batchSizeMult' iterations.")
Args.add_argument("--optimzer", type=str, help="")
Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
Args.add_argument("--momentum", type=float, help="SGD Momentum.")

class SGDW(torch.optim.SGD):
    """ decouples weight decay from lr in SGD (https://arxiv.org/pdf/1711.05101.pdf)
    """
    def __init__(self, params, weight_decay=2e-4, **kwargs):
        defaults = dict(weight_decay=0,
                        **kwargs)
        super(SGDW, self).__init__(params, **defaults)
        
        # add 'weight_decay_decoupled' manually
        self.defaults.update(weight_decay_decoupled=weight_decay)
        for group in self.param_groups:
            group['weight_decay_decoupled'] = weight_decay

    def step(self):
        """
        run normal sgd with wd=0 and apply decoupled wd afterwards
        """
        super(SGDW, self).step()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.add_(p, alpha=-group['weight_decay_decoupled'])


class Optimizer(SGDW):
    def __init__(self, params, **kwargs):
        defaults = dict(lr           = Args.learningRate,
                        momentum     = Args.momentum,
                        weight_decay = Args.weightDecay,
                        nesterov     = Args.nesterov,
                        **kwargs)
        super(Optimizer, self).__init__(params, **defaults)
        self.batchSizeMult = Args.batchSizeMult
        self.batchCounter = 0

    @torch.no_grad()
    def step(self):
        if self.batchSizeMult == 1:
            #handle == 1 case separately for better performance 
            super(Optimizer, self).step()
            self.zero_grad()
        else:
            self.batchCounter += 1
            if self.batchCounter % self.batchSizeMult == 0:
                self.batchCounter = 0

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.grad.div_(self.batchSizeMult)

                super(Optimizer, self).step()
                self.zero_grad()


    def load_state_dict(self, state_dict) -> None:
        super(Optimizer, self).load_state_dict(state_dict)
        
        #set parameters to Args parameters, otherwise parameters from state dictionary will stay in use
        self.defaults["lr"] = Args.learningRate
        self.defaults["momentum"] = Args.momentum
        self.defaults["weight_decay"] = Args.weightDecay
        self.defaults["nesterov"] = Args.nesterov

        for group in self.param_groups:
            for name, item in self.defaults.items():
                group[name] = item


    def reset_momentum_buffer(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'momentum_buffer' in state:
                    del state["momentum_buffer"]

