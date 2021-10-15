import os
import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool   (name = "saveCheckpoint"        , default = False     , help = "save model after each epoch (delete after next epoch)")
flags.DEFINE_bool   (name = "saveBestModel"         , default = False     , help = "keep best model")
flags.DEFINE_bool   (name = "keepLastCheckpoint"    , default = False     , help = "keep checkpoint after last epoch is done")
flags.DEFINE_list   (name = "checkpointsList"       , default = []        , help = "keep checkpoint at specific epochs")
flags.DEFINE_integer(name = "saveCheckpointInterval", default = 0         , help = "save model between interval")

class ModelSaver():
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            self.model = model
        else:
            self.model = model.module

        self.optimizer = optimizer
        self.dir = os.path.join(FLAGS.logDir, FLAGS.logSubDir)
        self.bestTestAccur = 0

    def __call__(self, epoch, testAccur):
        if torch.distributed.get_rank() == 0:
            if epoch in map(int, FLAGS.checkpointsList) or \
                FLAGS.saveCheckpoint or \
                (FLAGS.saveBestModel and self.bestTestAccur < testAccur):
                
                state = {'epoch': epoch,
                        'modelState': self.model.state_dict(),
                        'optimizerState': self.optimizer.state_dict(),
                        'bestTestAccur': self.bestTestAccur,
                        }

                if epoch in map(int, FLAGS.checkpointsList):
                    torch.save(state, os.path.join(self.dir, f"epoch_{epoch}.model"))
                if FLAGS.saveCheckpoint:
                    if epoch == FLAGS.epochs - 1 and not FLAGS.keepLastCheckpoint:
                        os.remove(os.path.join(self.dir, f"checkpoint.model"))
                    elif epoch % FLAGS.saveCheckpointInterval and FLAGS.saveCheckpointInterval > 0:
                        torch.save(state, os.path.join(self.dir, f"checkpoint_{epoch}.model"))
                    else:
                        torch.save(state, os.path.join(self.dir, f"checkpoint.model"))
                if FLAGS.saveBestModel and self.bestTestAccur < testAccur:
                    self.bestTestAccur = testAccur
                    torch.save(state, os.path.join(self.dir, f"bestModel.model"))

    def loadLast(self):
        lastEpoch = -1
        for file in os.listdir(self.dir):
            if file.startswith("epoch_") and file.endswith(".model"):
                epoch = int(file[6:-6])
                lastEpoch = epoch if epoch > lastEpoch else lastEpoch
        if lastEpoch == -1:
            raise RuntimeError(f"No epoch checkpoint found in dir: {self.dir}")

        return self.loadModel(filename = f"epoch_{lastEpoch}.model")


    def loadModel(self, filename):
        #@TODO save and load rnd state
        state = torch.load(os.path.join(self.dir, filename),  map_location='cpu')

        print(f"Loading from epoch {state['epoch']} (File: {filename}).")

        self.model.load_state_dict(state['modelState'])
        self.optimizer.load_state_dict(state['optimizerState'])

        self.bestTestAccur = state['bestTestAccur']

        return state['epoch']
