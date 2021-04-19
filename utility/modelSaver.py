import shutil
import os
import torch

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool   (name = "saveCheckpoint"    , default = False        , help = "save model after each epoch (delete after next epoch)")
flags.DEFINE_bool   (name = "saveBestModel"     , default = False        , help = "keep best model")
flags.DEFINE_bool   (name = "keepLastCheckpoint", default = False        , help = "keep checkpoint after last epoch is done")
flags.DEFINE_list   (name = "checkpointsList"   , default = []        , help = "keep checkpoint at specific epochs")


class ModelSaver():
    def __init__(self, model, optimizer):
        self.model     = model
        self.optimizer = optimizer
        self.dir = os.path.join(FLAGS.logDir, FLAGS.logSubDir)
        self.bestTestAccur = 0
        pass

    def __call__(self, epoch, testAccur):
        if epoch in [int(ep) for ep in FLAGS.checkpointsList] or \
            FLAGS.saveCheckpoint or \
            (FLAGS.saveBestModel and self.bestTestAccur < testAccur):
            
            state = {'epoch': epoch,
                     'modelState': self.model.state_dict(),
                     'optimizerState': self.optimizer.state_dict(),
                     'bestTestAccur': self.bestTestAccur,
                     }

            if epoch in [int(ep) for ep in FLAGS.checkpointsList]:
                torch.save(state, os.path.join(self.dir, f"epoch_{epoch}.model"))
            if FLAGS.saveCheckpoint:
                if epoch == FLAGS.epochs - 1 and not FLAGS.keepLastCheckpoint:
                    os.remove(os.path.join(self.dir, f"checkpoint.model"))
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
        state = torch.load(os.path.join(self.dir, filename))

        print(f"Loading from epoch {state['epoch']} (File: {filename}).")

        self.model.load_state_dict(state['modelState'])
        self.optimizer.load_state_dict(state['optimizerState'])

        self.bestTestAccur = state['bestTestAccur']

        return state['epoch']
