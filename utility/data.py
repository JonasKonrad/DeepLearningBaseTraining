import torch
import torchvision
import torchvision.transforms as transforms
import os
from .augmentation import AutoAugment, cutout
import numpy as np
import random

from absl import flags

def worker_init_fn(id):
    """ manually seed each workers np random generator. see https://github.com/pytorch/pytorch/issues/5059"""
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, train = True, download = None, transform = []):
        if train:
            dir = os.path.join(root, 'train')
            transform.transforms = [
                transforms.RandomResizedCrop(FLAGS.imageSize if FLAGS.imageSize is not None else 224, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3.))
                ] + transform.transforms
        else:
            dir = os.path.join(root, 'val')
            transform.transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                ] + transform.transforms
            if FLAGS.imageSize is not None:
                transform.transforms.insert(2, transforms.Resize(FLAGS.imageSize))

        super(ImageNet, self).__init__(dir, transform = transform)

# name, numClasses
availableDatasets = {
    "ImageNet": [ImageNet, 1000],
    "CIFAR10" : [torchvision.datasets.CIFAR10 , 10],
    "CIFAR100": [torchvision.datasets.CIFAR100, 100],
}
dataSetStatistics = {
    "ImageNet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "CIFAR10" : [[0.4913999140262604, 0.48215872049331665, 0.4465313255786896], [0.24703197181224823, 0.243484228849411, 0.26158687472343445]],
    "CIFAR100": [[0.5070753693580627, 0.4865487813949585, 0.44091784954071045], [0.2673334777355194, 0.25643861293792725, 0.2761503756046295]],
}


FLAGS = flags.FLAGS
flags.DEFINE_integer(name = "dataThreads" , default = 6            , help = "Number of CPU threads for dataloaders.")
flags.DEFINE_string (name = 'dataDir'     , default = "~/.datasets", help = "main directory to store datasets")
flags.DEFINE_integer(name = 'batchSize'   , default = 256          , help = "batch size")
flags.DEFINE_enum   (name = "dataset"     , default = "CIFAR100"   , enum_values = availableDatasets.keys(), help="Dataset")


flags.DEFINE_integer(name = 'imageSize'   , default = None          , help = "resize images after augmentation")


flags.DEFINE_bool   (name = "flip"       , default = False        , help = "flip horizontally")
flags.DEFINE_bool   (name = "crop"       , default = False        , help = "crop 32x32 padding 4")
flags.DEFINE_bool   (name = "cut"        , default = False        , help = "cutout")
flags.DEFINE_float  (name = "cutoutProp" , default = 0.5          , help = "Probability for cutout augmenation.")
flags.DEFINE_bool   (name = "autoAugment", default = False        , help = "autoAugment")


class DataLoader:
    def __init__(self, num_replicas = 1, rank = 1):
        self.datasetName = FLAGS.dataset
        try:
            self.dataset, self.numClasses = availableDatasets[self.datasetName]
        except KeyError:
            raise NameError(f"Dataset {self.datasetName} not found. Available datasets are: {', '.join(availableDatasets.keys())}")


        if self.datasetName in dataSetStatistics:
            mean, std = dataSetStatistics[self.datasetName]
        else:
            print("Calculating Mean and Std...")
            mean, std = self._get_statistics()
            print(f"Mean = {list(map(float,mean))}, Std = {list(map(float,std))}")


        transform_list = []
        if FLAGS.autoAugment and self.datasetName in ["CIFAR10","CIFAR100"]:
            transform_list.append(AutoAugment())
        if FLAGS.flip:
            transform_list.append(torchvision.transforms.RandomHorizontalFlip())
        if FLAGS.cut:
            transform_list.append(Cutout())
        if FLAGS.crop and self.datasetName in ["CIFAR10","CIFAR100"]:
            transform_list.append(torchvision.transforms.RandomCrop(size=(32, 32), padding=4))

        if FLAGS.imageSize is not None:
            transform_list += [
                transforms.Resize(FLAGS.imageSize, interpolation = torchvision.transforms.InterpolationMode.BICUBIC),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        train_transform = transforms.Compose(transform_list)

        if FLAGS.imageSize is not None:
            test_transform = [
                transforms.Resize(FLAGS.imageSize, interpolation = torchvision.transforms.InterpolationMode.BICUBIC),
            ]
        else:
            test_transform = []

        test_transform = transforms.Compose(test_transform + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = self.dataset(root=FLAGS.dataDir, train=True, download=True, transform=train_transform)
        test_set  = self.dataset(root=FLAGS.dataDir, train=False, download=True, transform=test_transform)
        
        # to get random order of train data the seed has to be set manually. the seed value must be constant among all processes. 
        seed = torch.tensor(random.getrandbits(32)).cuda(FLAGS.local_rank % torch.cuda.device_count())
        torch.distributed.broadcast(seed, src = 0)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True , seed = seed.data)
        test_sampler  = torch.utils.data.distributed.DistributedSampler(test_set , shuffle=False)
        
        self.train = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batchSize, num_workers=FLAGS.dataThreads, worker_init_fn=worker_init_fn, sampler = train_sampler)
        self.test  = torch.utils.data.DataLoader(test_set , batch_size=FLAGS.batchSize, num_workers=FLAGS.dataThreads, worker_init_fn=worker_init_fn, sampler = test_sampler)

    def _get_statistics(self):
        data = self.dataset(root=FLAGS.dataDir, train=True, download=True, transform=transforms.ToTensor())
        firstMoment = torch.zeros(3)
        secondMoment = torch.zeros(3)
        N = 0
        for inputs, _ in torch.utils.data.DataLoader(data, batch_size=64, num_workers=FLAGS.dataThreads):
            N += inputs.shape[0]
            for i in range(3):
                firstMoment[i] += inputs[:,i,:,:].mean()*inputs.shape[0]
                secondMoment[i] += (inputs[:,i,:,:]**2).mean()*inputs.shape[0]
        firstMoment .div_(N)
        secondMoment.div_(N)
        return firstMoment, torch.sqrt(secondMoment-firstMoment**2)


class Cutout:
    def __init__(self, size=16):
        self.size = size
        self.half_size = size // 2
        self.p = FLAGS.cutoutProp

    def __call__(self, image):
        return cutout(image)
