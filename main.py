import os
import sys
import torch
import json
import torch.multiprocessing as mp
import torch.distributed as dist

from absl import flags
from absl import app

from models import modelDict
from utility.loss import smooth_crossentropy
from utility.data import DataLoader
from utility.log import Log
from utility.utils import initialize
from utility.LRScheduler import getLRScheduler, _LRScheduler
from utility.optimizer import SGD
from utility.modelSaver import ModelSaver

"""
@TODO:
    - add model summary
    - check correct rnd seeding; save rnd seed
"""

FLAGS = flags.FLAGS
app.define_help_flags()
flags.DEFINE_string (name = "logDir"      , default = "../logs"    , help = "main directory to store logs")
flags.DEFINE_string (name = "logSubDir"   , default = "test"       , help = "subdir in logDir to store logs for this run")
flags.DEFINE_integer(name = "epochs"      , default = 400          , help = "Total number of epochs.")
flags.DEFINE_bool   (name = "contin"      , default = False        , help = "Whether to set rnd seed.")
flags.DEFINE_bool   (name = "freezeBN"    , default = False        , help = "Whether to freezeBN.")

flags.DEFINE_integer(name = "gpus"       , default = 1          , help = "gpus per node. catched form 'GPUS_PER_NODE' if started with slurm")
flags.DEFINE_integer(name = "local_rank" , default = 0         , help = "local process rank. catched form 'SLURM_PROCID' if started with slurm")
flags.DEFINE_integer(name = "nodes"      , default = 1          , help = "")
flags.DEFINE_integer(name = "world_size" , default = 1          , help = "")


def train() -> None:
    log = Log(logDir = logDir)

    dist.init_process_group(backend="nccl", init_method="env://", world_size=FLAGS.world_size, rank=FLAGS.local_rank)

    localGPU = FLAGS.local_rank % FLAGS.gpus
    torch.cuda.set_device(localGPU)

    dataset = DataLoader(num_replicas = 1, rank = 1)

    model: torch.nn.Module = modelDict[FLAGS.model](num_classes=dataset.numClasses)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda(localGPU)


    optimizer = SGD(model.parameters())
    # optimizer: torch.optim.Optimizer = optimizerDict[FLAGS.model](num_classes=dataset.numClasses)
    lrScheduler: _LRScheduler = getLRScheduler(optimizer)
    modelSaver = ModelSaver(model = model, optimizer = optimizer)
    
    startEpoch = 1
    if FLAGS.contin:
        startEpoch = modelSaver.loadModel("checkpoint.model")
        startEpoch += 1
        if startEpoch >= FLAGS.epochs:
            raise RuntimeError(f"Can't cotinue model from epoch {startEpoch} to max epoch {FLAGS.epochs}.")

    for epoch in range(startEpoch, FLAGS.epochs+1):
        dataset.train.sampler.set_epoch(epoch)

        model.train()
        numBatches = len(dataset.train)
        log.train(epoch, len_dataset=numBatches)

        if FLAGS.freezeBN:
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
    FLAGS(sys.argv)

    logDir = os.path.join(FLAGS.logDir, FLAGS.logSubDir)

    if FLAGS.contin:
        #first overwrite defaults with old parameters ..
        with open(os.path.join(logDir, "params.json"), "r") as file:
            flagsDict = json.load(file)
        FLAGS._set_attributes(**flagsDict)
        #@TODO add validation of flags here
        #... then set overwrite old parameters with new parameters
        FLAGS(sys.argv)

    initialize() #set up seed and cudnn

    FLAGS.local_rank = os.getenv("SLURM_PROCID", FLAGS.local_rank)
    FLAGS.gpus       = os.getenv("GPUS_PER_NODE", FLAGS.gpus)
    FLAGS.nodes      = os.getenv("SLURM_JOB_NUM_NODES", 1)
    FLAGS.world_size = FLAGS.gpus * FLAGS.nodes

    if FLAGS.local_rank == 0:
        os.makedirs(logDir, exist_ok=True)
        with open(os.path.join(logDir, "params.json"), "w") as file:
            json.dump(FLAGS.flag_values_dict(), file, indent = 4)

    train()