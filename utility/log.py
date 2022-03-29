import h5py
import time
import torch
import os

from utility.args import Args


"""
@TODO:
    - make learning rate no special variable (just normal metric) anymore
    - add option to metric at each batch
    - implement buffer/flush
    - slim down this code...
"""


Args.add_argument("--truncate", type=bool, help="truncate log file")
Args.add_argument("--verbose", type=bool, help="print to terminal")
Args.add_argument("--logEach", type=int, help="Iterations to log during training.")

class Log:
    def __init__(self, metricConfig = [], logDir = "logs"):
        """
        metricConfig: list of dicts. each dict represents one metric. dict can contain args: name, logTrain, logTest, showTrain, showTest, fmt
        """
        self.commandLine = Args.verbose
        self.logEach = Args.logEach

        self.columnLen = 15
        self.epoch = -1
        self.batches = 0
        self.steps = 0
        self.state = {}

        self.filePath = os.path.join(Args.logDir, Args.logSubDir, "logs.hdf5")
        if torch.distributed.get_rank() == 0:
            if not Args.contin:
                with h5py.File(self.filePath, "w" if Args.truncate else "w-") as f:
                    f.create_group("train")
                    f.create_group("test")
                    f["train"].create_dataset("LR", shape=(0,), dtype=float, maxshape=(Args.epochs,), chunks=True)

            else:
                if not os.path.isfile(self.filePath):
                    raise RuntimeError(f"Running in continue mode but log file not found. Path: {self.filePath}")
        
        self.config = {}
        self.defaultConfig = {
            "logTrain"  : True,
            "logTest"   : True,
            "showTrain" : False,
            "showTest"  : False,
            "fmt": lambda x: f"{x:.4f}"
        }


        self._addMetric({
            "name"      : "loss",
            "showTrain" : True,
            "showTest"  : True,
            })
        self._addMetric({
            "name"      : "accuracy",
            "showTrain" : True,
            "showTest"  : True,
            "fmt": lambda x: f"{100*x:.2f} %"
            })
        for conf in metricConfig:
            self._addMetric(conf)


        self.showMetricsTrain = []
        self.showMetricsTest  = []
        for name, conf in self.config.items():
            if conf["showTrain"]:
                self.showMetricsTrain.append(name)
            if conf["showTest"]:
                self.showMetricsTest.append(name)

        self.loading_bar = LoadingBar(length=(self.columnLen+1)*len(self.showMetricsTest)-3)

    def train(self, epoch, len_dataset: int) -> None:
        """ reset to be ready for eval data """
        if torch.distributed.get_rank() == 0:
            self.epoch = epoch

            self.is_train = True
            self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        """ write train data and reset to be ready for eval data """
        if torch.distributed.get_rank() == 0:
            self.flush()
            self.is_train = False

            with h5py.File(self.filePath, "r+") as f:
                fileGroup = f["train"]
                for name, val in self.state.items():
                    fileGroup[name].resize(self.epoch, axis = 0)
                    fileGroup[name][..., -1] = val / self.steps

                fileGroup["LR"].resize(self.epoch, axis = 0)
                fileGroup["LR"][..., -1] = self.learning_rate

            self._reset(len_dataset)

    def evalEnd(self) -> None:
        """ write eval data """
        if torch.distributed.get_rank() == 0:
            #print new line
            self.flush()
            print()

            with h5py.File(self.filePath, "r+") as f:
                fileGroup = f["test"]
                for name, val in self.state.items():
                    fileGroup[name].resize(self.epoch, axis = 0)
                    fileGroup[name][..., -1] = val / self.steps

    def __call__(self, logs, learning_rate: float = None) -> None:
        self.learning_rate = learning_rate
        for key, log in logs.items():
            if key not in self.state:
                self._addMetric({"name": key})

            logSum = log.sum()
            torch.distributed.reduce(logSum, 0)
            if torch.distributed.get_rank() == 0:
                self.state[key] += logSum.item()

        newSteps = torch.Tensor([logs["loss"].size(0)]).cuda(torch.distributed.get_rank() % torch.cuda.device_count())
        torch.distributed.reduce(newSteps, 0)

        if torch.distributed.get_rank() == 0:
            self.steps   += newSteps.item()
            self.batches += 1
            if self.commandLine and self.batches % self.logEach == self.logEach - 1:
                self.flush()

    def getScalar(self, name):
        return self.state[name] / self.steps if self.steps != 0 else None

    def flush(self) -> None:
        """ print to terminal """
        if self.is_train:
            self.trainString = f"{f'{self.learning_rate:.3e}'.center(self.columnLen-1)}┃{'│'.join([self.config[name]['fmt'](self.state[name] / self.steps).center(self.columnLen) for name in self.showMetricsTrain])}"
            
            if self.commandLine:
                print(f"\r┃{str(self.epoch).center(self.columnLen-1)}┃{self._time().center(self.columnLen-1)}│{self.trainString}{self.loading_bar(self.batches / self.len_dataset)}",
                    end="",
                    flush=True)
        else:
            start = '\r' if self.commandLine else ''
            print(f"{start}┃{str(self.epoch).center(self.columnLen-1)}┃{self._time().center(self.columnLen-1)}│{self.trainString}┃{'│'.join([self.config[name]['fmt'](self.state[name] / self.steps).center(self.columnLen) for name in self.showMetricsTest])}┃",
                end="")

            
    def _reset(self, len_dataset: int) -> None:
        if self.is_train:
            self.start_time = time.time()
        self.steps = 0
        self.batches = 0
        self.len_dataset = len_dataset
        for key in self.state:
            self.state[key] = 0

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _addMetric(self, config):
        name = config["name"]
        internConfig = {}
        for key, val in self.defaultConfig.items():
            if key in config:
                internConfig[key] = config[key]
            else:
                internConfig[key] = val
        self.config[name] = internConfig
        self.state[name] = 0

        if torch.distributed.get_rank() == 0:
            if not Args.contin:
                with h5py.File(self.filePath, "r+") as f:
                    if internConfig["logTrain"]:
                        f["train"].create_dataset(name, shape=(0,), dtype=float, maxshape=(Args.epochs,), chunks=True)
                    if internConfig["logTest"]:
                        f["test"].create_dataset(name, shape=(0,), dtype=float, maxshape=(Args.epochs,), chunks=True)

    def print_header(self) -> None:
        if torch.distributed.get_rank() == 0:
            print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳{'T╺╸R╺╸A╺╸I╺╸N '.center((self.columnLen+1)*len(self.showMetricsTrain)-1,'━')}┳{'T╺╸E╺╸S╺╸T '.center((self.columnLen+1)*len(self.showMetricsTest)-1,'━')}┓")
            print(f"┃              ┃                             ┃{' '*((self.columnLen+1)*len(self.showMetricsTrain)-1)}┃{' '*((self.columnLen+1)*len(self.showMetricsTest)-1)}┃")
            print(f"┃    epoch     ┃      time    │     l.r.     ┃{'│'.join([name[:self.columnLen].center(self.columnLen) for name in self.showMetricsTrain])}┃{'│'.join([name[:self.columnLen].center(self.columnLen) for name in self.showMetricsTest])}┃")
            print(f"┠──────────────╂──────────────┼──────────────╂{'┼'.join(['─'*self.columnLen]*len(self.showMetricsTrain))}╂{'┼'.join(['─'*self.columnLen]*len(self.showMetricsTest))}┨")

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"
