from utility.loading_bar import LoadingBar
from torch.utils.tensorboard import SummaryWriter
import time

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(name = "logEach"     , default = 1            , help = "Iterations to log during training.")

class Log:
    def __init__(self, log_each: int, initial_epoch=-1, logDir = "logs"):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.epoch_state = {"steps": 0}

        self.writer = SummaryWriter(logDir)
    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False

        for key, value in self.epoch_state.items():
            self.writer.add_scalar(f'{key}/train', self.epoch_state[key] / self.epoch_state["steps"], self.epoch)
        self.writer.flush()

        self._reset(len_dataset)

    def evalEnd(self) -> None:
        self.writer.add_scalar('loss/test', self.epoch_state["loss"] / self.epoch_state["steps"], self.epoch)
        self.writer.add_scalar('accuracy/test', self.epoch_state["accuracy"] / self.epoch_state["steps"], self.epoch)
        self.writer.flush()

    def __call__(self, logs, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(logs, learning_rate)
        else:
            self._eval_step(logs)

    def flush(self) -> None:
        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )
        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(f"{loss:12.4f}  │{100*accuracy:10.2f} %  ┃", flush=True)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

    def _train_step(self, logs, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        for key, log in logs.items():
            if key in self.epoch_state: self.epoch_state[key] += log.sum().item()
            else:                       self.epoch_state[key]  = log.sum().item()

        self.epoch_state["steps"] += logs["loss"].size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]


            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, logs) -> None:
        for keys, log in logs.items():
            self.epoch_state[keys] += log.sum().item()
        self.epoch_state["steps"] += logs["loss"].size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        for key, _ in self.epoch_state.items():
            self.epoch_state[key] = 0

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━T╺╸R╺╸A╺╸I╺╸N ━━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃                             ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")
