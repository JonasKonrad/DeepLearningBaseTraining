from torch.utils.tensorboard import SummaryWriter
import time
import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool   (name = "verbose"     , default = False        , help = "print to terminal")
flags.DEFINE_integer(name = "logEach"     , default = 1            , help = "Iterations to log during training.")

class Log:
    def __init__(self, metricConfig = [], logDir = "logs"):
        """
        metricConfig: list of dicts. each dict represents one metric. dict can contain args: name, logTrain, logTest, showTrain, showTest, fmt
        """
        self.commandLine = FLAGS.verbose
        self.logEach = FLAGS.logEach

        self.columnLen = 15
        self.epoch = -1
        self.batches = 0
        self.steps = 0
        self.state = {}

        self.writer = SummaryWriter(logDir)
    
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
        if torch.distributed.get_rank() == 0:
            self.epoch = epoch
            if self.epoch == 1:
                self._print_header()

            self.is_train = True
            self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        if torch.distributed.get_rank() == 0:
            self.flush()
            self.is_train = False

            for name, val in self.state.items():
                if self.config[name]["logTrain"]:
                    self.writer.add_scalar(f'{name}/train', val / self.steps, self.epoch)
            self.writer.add_scalar('LR',self.learning_rate, self.epoch)
            self.writer.flush()

            self._reset(len_dataset)

    def evalEnd(self) -> None:
        if torch.distributed.get_rank() == 0:
            #print new line
            self.flush()
            print()
            for name, val in self.state.items():
                if self.config[name]["logTest"]:
                    self.writer.add_scalar(f'{name}/test', val / self.steps, self.epoch)
            self.writer.flush()

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
        print(f"reset! train: {self.is_train}. steps processed: {self.steps}")
        self.steps = 0
        self.batches = 0
        self.len_dataset = len_dataset
        for key in self.state:
            self.state[key] = 0

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _addMetric(self, config):
        internConfig = {}
        for key, val in self.defaultConfig.items():
            if key in config:
                internConfig[key] = config[key]
            else:
                internConfig[key] = val
        self.config[config["name"]] = internConfig
        self.state[config["name"]] = 0

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳{'T╺╸R╺╸A╺╸I╺╸N '.center((self.columnLen+1)*len(self.showMetricsTrain)-1,'━')}┳{'V╺╸A╺╸L╺╸I╺╸D '.center((self.columnLen+1)*len(self.showMetricsTest)-1,'━')}┓")
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
