import torch

from .sgd import SGD
from .adamW import AdamW


from utility.args import Args
Args.add_argument("--optimizer", type=str, help="optimizer name")
Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
Args.add_argument("--momentum", type=float, help="Momentum.")
Args.add_argument("--nesterov", type=bool, help="use normal nesterov for sgd")

Args.add_argument("--grad_clip_norm", type=float, help="")


optimizerDict = {
    "SGD": SGD,
    "AdamW": AdamW,
}


def getOptimizer() -> torch.optim.Optimizer:
    if Args.optimizer in optimizerDict:
        return optimizerDict[Args.optimizer]
    else:
        raise RuntimeError(f"Optimizer '{Args.optimizer}' not found. Available optimizers: {', '.join(optimizerDict.keys())}")
