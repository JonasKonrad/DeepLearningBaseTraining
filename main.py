import os
import torch
import json

from models import getModel
from optimizer import getOptimizer
from utility.loss import smooth_crossentropy
from utility.data import DataLoader
from utility.dataLogger import DataLogger
from utility.utils import initialize
from utility.LRScheduler import getLRScheduler, _LRScheduler
from utility.modelSaver import ModelSaver
from utility.args import Args


"""
run:
    python -m torch.distributed.run main.py
"""

Args.add_argument("--logDir", type=str, help="main directory to store logs")
Args.add_argument("--logSubDir", type=str, help="subdir in logDir to store logs for this run")
Args.add_argument("--epochs", type=int, help="Total number of epochs")
Args.add_argument("--contin", type=bool, help="Whether to continue from checkpoint. In continue mode parameters are read from params.json file, input file is ignored.")
Args.add_argument("--local_rank", type=int, help="local process rank. catched form 'SLURM_PROCID' if started with slurm")


def train() -> None:
    torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=Args.local_rank)
    localGPU = Args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(localGPU)

    dataLogger = DataLogger()
    dataset = DataLoader()

    model = getModel()(num_classes=dataset.numClasses)
    model = model.cuda(localGPU)
    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)


    optimizer = getOptimizer()(model.parameters(), model.named_parameters())
    lrScheduler: _LRScheduler = getLRScheduler(optimizer)
    modelSaver = ModelSaver(model = model, optimizer = optimizer)
    
    startEpoch = 1
    if Args.contin:
        startEpoch = modelSaver.loadModel("checkpoint.model")
        startEpoch += 1
        model = model.cuda(localGPU)
        if startEpoch >= Args.epochs:
            raise RuntimeError(f"Can't continue model from epoch {startEpoch} to max epoch {Args.epochs}.")
    else:
        modelSaver(0, 0)

    dataLogger.printHeader()
    state = {
        "model": model,
        "lrScheduler": lrScheduler,
        "optimizer": optimizer,
    }
    for epoch in range(startEpoch, Args.epochs+1):
        dataset.train.sampler.set_epoch(epoch)

        model.train()
        numBatches = len(dataset.train)
        dataLogger.startTrain(trainDataLen = numBatches)

        for i, batch in enumerate(dataset.train):
            inputs, targets = (b.cuda(localGPU) for b in batch)

            predictions = model(inputs)

            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            
            if Args.grad_clip_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            lrScheduler.step(epoch-1, (i+1)/numBatches)

            with torch.no_grad():
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

        dataLogger.flush()
        
        dataLogger.startTest()
        model.eval()
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda(localGPU) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

            dataLogger.flush()
            modelSaver(epoch, 1) #@TODO log.getScalar("accuracy")



if __name__ == "__main__":
    Args.parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    logDir = os.path.join(Args.logDir, Args.logSubDir)

    if Args.contin:
        with open(os.path.join(logDir, "params.json"), "r") as file:
            parameters = json.load(file)
        Args.parse_args_contin(parameters)

    Args.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_PROCID", Args.local_rank)))
    initialize() # set up seed and cudnn

    if Args.local_rank == 0:
        os.makedirs(logDir, exist_ok=True)
        with open(os.path.join(logDir, "params.json"), "w") as file:
            json.dump(vars(Args.data), file, indent = 4)

    train()
