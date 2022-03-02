import os
import sys
import torch
import json

from models import modelDict
from utility.loss import smooth_crossentropy
from utility.data import DataLoader
from utility.log import Log
from utility.utils import initialize
from utility.LRScheduler import getLRScheduler, _LRScheduler
from utility.optimizer import SGD
from utility.modelSaver import ModelSaver
from utility.args import Args


"""
run:
    python -m torch.distributed.run main.py
"""

"""
@TODO:
    - add model summary
    - save rnd seed
    - typing...
    - reintroduce cpu support
    - implement PEP 8
    - update docstrings
"""


Args.add_argument("--logDir", type=str, help="main directory to store logs")
Args.add_argument("--logSubDir", type=str, help="subdir in logDir to store logs for this run")
Args.add_argument("--epochs", type=int, help="Total number of epochs")
Args.add_argument("--contin", type=bool, help="Whether to continue from checkpoint. In continue mode parameters are read from params.json file, input file is ignored.")
Args.add_argument("--freezeBN", type=bool, help="Whether to freezeBN.")

Args.add_argument("--local_rank", type=int, help="local process rank. catched form 'SLURM_PROCID' if started with slurm")
Args.add_argument("--nodes", type=int, help="")
Args.add_argument("--world_size", type=int, help="")


def train() -> None:
    logDir = os.path.join(Args.logDir, Args.logSubDir)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=Args.world_size, rank=Args.local_rank)
    localGPU = Args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(localGPU)

    log = Log(logDir = logDir)

    dataset = DataLoader(num_replicas = 1, rank = 1)

    model: torch.nn.Module = modelDict[Args.model](num_classes=dataset.numClasses)
    model = model.cuda(localGPU)
    if not Args.freezeBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)


    optimizer = SGD(model.parameters())
    # optimizer: torch.optim.Optimizer = optimizerDict[Args.model](num_classes=dataset.numClasses)
    lrScheduler: _LRScheduler = getLRScheduler(optimizer)
    modelSaver = ModelSaver(model = model, optimizer = optimizer)
    
    startEpoch = 1
    if Args.contin:
        startEpoch = modelSaver.loadModel("checkpoint.model")
        startEpoch += 1
        model = model.cuda(localGPU)
        log._print_header()
        if startEpoch >= Args.epochs:
            raise RuntimeError(f"Can't continue model from epoch {startEpoch} to max epoch {Args.epochs}.")
    else:
        modelSaver(0, 0)

    for epoch in range(startEpoch, Args.epochs+1):
        dataset.train.sampler.set_epoch(epoch)

        model.train()
        numBatches = len(dataset.train)
        log.train(epoch, len_dataset=numBatches)

        if Args.freezeBN:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.training = False
                    m.track_running_stats = False

        for i, batch in enumerate(dataset.train):
            inputs, targets = (b.cuda(localGPU) for b in batch)

            predictions = model(inputs)

            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            logs = {"loss": loss}
            
            optimizer.step()

            lrScheduler.step(epoch-1, (i+1)/numBatches)

            with torch.no_grad():
                logs["accuracy"] = torch.argmax(predictions.data, 1) == targets
                log(logs, learning_rate = lrScheduler.get_last_lr()[0])

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda(localGPU) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log({"loss": loss, "accuracy": correct})

            log.evalEnd()
        
            modelSaver(epoch, log.getScalar("accuracy"))



if __name__ == "__main__":
    Args.parse_args()

    logDir = os.path.join(Args.logDir, Args.logSubDir)

    if Args.contin:
        with open(os.path.join(logDir, "params.json"), "r") as file:
            parameters = json.load(file)
        Args.parse_args_contin(parameters)

    initialize() # set up seed and cudnn

    Args.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_PROCID", Args.local_rank)))
    Args.nodes      = int(os.getenv("SLURM_JOB_NUM_NODES", Args.nodes))
    Args.world_size = torch.cuda.device_count() * Args.nodes

    if Args.local_rank == 0:
        os.makedirs(logDir, exist_ok=True)
        with open(os.path.join(logDir, "params.json"), "w") as file:
            json.dump(vars(Args.data), file, indent = 4)

    train()
