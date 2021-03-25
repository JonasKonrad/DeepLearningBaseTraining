import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import json

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logDir"         , default="../logs"   , type=str, help = "main directory to store logs")
    parser.add_argument("--logSubDir"      , default="test"   , type=str, help = "subdir in logDir to store logs for this run")
    parser.add_argument("--batch_size"     , default=128      , type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth"          , default=28       , type=int, help="Number of layers.")
    parser.add_argument("--width_factor"   , default=10       , type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--dataset"        , default="CIFAR100", choices=["CIFAR10", "CIFAR100"], help="Dataset")
    parser.add_argument("--dropout"        , default=0.0      , type=float, help="Dropout rate.")
    parser.add_argument("--epochs"         , default=300      , type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate"  , default=0.1      , type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum"       , default=0.9      , type=float, help="SGD Momentum.")
    parser.add_argument("--threads"        , default=2        , type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--logEach"        , default=-1       , type=int)
    parser.add_argument("--weight_decay"   , default=0.0005   , type=float, help="L2 weight decay.")
    parser.add_argument("--rndSeed"        , action="store_true")
    args = parser.parse_args()


    logDir = os.path.join(args.logDir, args.logSubDir)
    os.makedirs(logDir, exist_ok=True)
    with open(os.path.join(logDir, "params.json"), "w") as file:
        json.dump(vars(args), file, indent = 4)

    if args.rndSeed: initialize(args)
    else           : initialize(args, seed=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    dataset = Cifar(args.batch_size, args.threads, dataset = args.dataset)
    log = Log(log_each=args.logEach, logDir = logDir)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=dataset.numClasses).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)

            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            logs = {"loss": loss}
            
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                logs["accuracy"] = torch.argmax(predictions.data, 1) == targets
                log({key: log.cpu() for key, log in logs.items()}, learning_rate = scheduler.lr())
                scheduler(epoch)
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
