import torch
import torchvision
import torchvision.transforms as transforms

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(name = "dataThreads" , default = 2            , help = "Number of CPU threads for dataloaders.")
flags.DEFINE_enum   (name = "dataset"     , default = "CIFAR100"   , enum_values = ["CIFAR10", "CIFAR100"], help="Dataset")
flags.DEFINE_string (name = 'dataDir'     , default = "~/.datasets", help = "main directory to store datasets")
flags.DEFINE_integer(name = 'batchSize'   , default = 256          , help = "batch size")


flags.DEFINE_bool   (name = "flip"     , default = False        , help = "flip horizontally")
flags.DEFINE_bool   (name = "crop"     , default = False        , help = "crop 32x32 padding 4")
flags.DEFINE_bool   (name = "cut"      , default = False        , help = "cutout")
flags.DEFINE_float(name = "cutoutProp" , default = 0.5, help = "Probability for cutout augmenation.")


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

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if FLAGS.flip: transform_list.append(torchvision.transforms.RandomHorizontalFlip())
        if FLAGS.crop: transform_list.append(torchvision.transforms.RandomCrop(size=(32, 32), padding=4))
        if FLAGS.cut : transform_list.append(Cutout())

        train_transform = transforms.Compose(transform_list)

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


class Cutout:
    def __init__(self, size=16):
        self.size = size
        self.half_size = size // 2
        self.p = FLAGS.cutoutProp

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image
