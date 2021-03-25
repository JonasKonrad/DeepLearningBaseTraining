import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import json

from absl import flags

from models.wide_res_net import WideResNet
from utility.smooth_cross_entropy import smooth_crossentropy
from utility.data import DataLoader
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.optimizer import SGD


FLAGS = flags.FLAGS
flags.DEFINE_string (name = "logDir"      , default = "../logs"    , help = "main directory to store logs")
flags.DEFINE_string (name = "logSubDir"   , default = "test"       , help = "subdir in logDir to store logs for this run")
flags.DEFINE_integer(name = "epochs"      , default = 300          , help = "Total number of epochs.")
flags.DEFINE_bool   (name = "rndSeed"     , default = False        , help = "Whether to set rnd seed.")

if __name__ == "__main__":
    FLAGS(sys.argv)

    logDir = os.path.join(FLAGS.logDir, FLAGS.logSubDir)
    os.makedirs(logDir, exist_ok=True)
    with open(os.path.join(logDir, "params.json"), "w") as file:
        json.dump(FLAGS.flag_values_dict(), file, indent = 4)

    if FLAGS.rndSeed: initialize(FLAGS)
    else           : initialize(FLAGS, seed=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    dataset = DataLoader()
    log = Log(log_each=FLAGS.logEach, logDir = logDir)
    model = WideResNet(in_channels=3, labels=dataset.numClasses).to(device)

    optimizer = SGD(model.parameters())
    LRscheduler = StepLR(optimizer, FLAGS.epochs)

    for epoch in range(FLAGS.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        LRscheduler(epoch)

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)

            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            logs = {"loss": loss}
            
            optimizer.step()

            with torch.no_grad():
                logs["accuracy"] = torch.argmax(predictions.data, 1) == targets
                log({key: log.cpu() for key, log in logs.items()}, learning_rate = LRscheduler.lr())
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log({"loss": loss.cpu(), "accuracy": correct.cpu()})

            log.evalEnd()



    log.flush()
