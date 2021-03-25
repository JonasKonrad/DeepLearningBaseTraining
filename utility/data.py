import torch
import torchvision
import torchvision.transforms as transforms

from utility.cutout import Cutout

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(name = "dataThreads" , default = 2            , help = "Number of CPU threads for dataloaders.")
flags.DEFINE_enum   (name = "dataset"     , default = "CIFAR100"   , enum_values = ["CIFAR10", "CIFAR100"], help="Dataset")
flags.DEFINE_string (name = 'dataDir'     , default = "~/.datasets", help = "main directory to store datasets")
flags.DEFINE_integer(name = 'batchSize'   , default = 128          , help = "batch size")

class DataLoader:
    def __init__(self):

        self.dataset = FLAGS.dataset

        if self.dataset == "CIFAR10":
            self.loader = torchvision.datasets.CIFAR10
            self.numClasses = 10
        elif self.dataset == "CIFAR100":
            self.loader = torchvision.datasets.CIFAR100
            self.numClasses = 100
        else:
            raise NameError(f"Dataset {self.dataset} not found.")


        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = self.loader(root=FLAGS.dataDir, train=True, download=True, transform=train_transform)
        test_set  = self.loader(root=FLAGS.dataDir, train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batchSize, shuffle=True , num_workers=FLAGS.dataThreads)
        self.test  = torch.utils.data.DataLoader(test_set , batch_size=FLAGS.batchSize, shuffle=False, num_workers=FLAGS.dataThreads)

    def _get_statistics(self):
        train_set = self.loader(root=FLAGS.dataDir, train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in torch.utils.data.DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
