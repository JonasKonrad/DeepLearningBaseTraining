import os
import sys
import torch
import json

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


    os.makedirs(logDir, exist_ok=True)
    with open(os.path.join(logDir, "params.json"), "w") as file:
        json.dump(FLAGS.flag_values_dict(), file, indent = 4)

    initialize() #set up seed and cudnn

    dataset = DataLoader()
    log = Log(logDir = logDir)

    model: torch.nn.Module = modelDict[FLAGS.model](num_classes=dataset.numClasses)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)
    model = model.to(device)

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
        model.train()
        numBatches = len(dataset.train)
        log.train(epoch, len_dataset=numBatches)

        if FLAGS.freezeBN:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.training = False
                    m.track_running_stats = False

        for i, batch in enumerate(dataset.train):
            inputs, targets = (b.to(device) for b in batch)

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
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log({"loss": loss, "accuracy": correct})

            log.evalEnd()
        
            modelSaver(epoch, log.getScalar("accuracy"))
