from util import Constants
import os
import numpy
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class MyDataset(Dataset):
    
    def __init__(self, size=100, number_features=20, overwrite=False):
        super(MyDataset).__init__()
        self.path = Constants.PATH_DATA / "MyDataset"
        try:
            os.makedirs(self.path)
        except:
            pass
        self.file_input = "input.npy"
        self.file_output = "output.npy"
        self.number_classes = 2
        self.size = size
        self.number_features = number_features
        self.num_workers = 4
        if overwrite:
            try:
                self.delete_data()
            except:
                pass
        self.create_data()
        self.inputs, self.outputs = self.read_data()
        self.samples = list(zip(self.inputs, self.outputs))
        
    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

    def create_data(self):
        file = self.path / self.file_input
        if file.is_file():
            return
        else:
            inputs = numpy.array([numpy.around(numpy.random.rand(self.number_features)) for i \
                                  in range(self.size)], dtype=numpy.float32)
            outputs = numpy.array([self.calculate_output(i) for i in inputs])
            numpy.save(self.path / self.file_input, inputs)
            numpy.save(self.path / self.file_output, outputs)
        
    def calculate_output(self, i):
        input_size = len(i)
        criteria = int(numpy.sum(i[int(input_size/4):int(input_size/2)]) >= numpy.sum(i[int(input_size/2):int(3*input_size/4)]))
        return numpy.array([criteria, 1 - criteria], dtype=numpy.float32)
    
    def read_data(self):
        return self.read_file(self.path / self.file_input), self.read_file(self.path / self.file_output)
    
    def read_file(self, path):
        return numpy.load(path)
    
    def delete_data(self):
        """ not compatible with kaggle """
        (self.path / self.file_input).unlink()
        (self.path / self.file_output).unlink()

    def get_sample(self):
        loader_sample = DataLoader(
            torch.utils.data.Subset(self, [0]),
            batch_size = 1,
            num_workers = 0,
            shuffle=False
        )
        for i, o in loader_sample:
            return (i, o)
        
    def get_train_test_loader(self, batchSize):
        self.number_classes = 2
        self.train_test_ratio = 0.8
        loader_train = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(0, int(self.train_test_ratio * self.size))),
                batch_size=batchSize,
                num_workers=self.num_workers
            )
        loader_test = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(int(self.train_test_ratio * self.size), self.size)),
                batch_size=batchSize,
                num_workers=self.num_workers
            )
        return (loader_train, loader_test)

class MyMNIST():
    def __init__(self, one_hot_target=True):
        self.path = Constants.PATH_DATA / "MNIST"
        try:
            os.makedirs(self.path)
        except:
            pass
        self.number_classes = 10
        self.number_features = 28 * 28
        self.one_hot_target = one_hot_target
        self.num_workers = 4
        self.dataset_train = torchvision.datasets.MNIST(
            self.path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=MyTargetTransform(self.one_hot_target)
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
            target_transform=MyTargetTransform(self.one_hot_target)
        )

    def get_sample(self):
        loader_sample = DataLoader(
            torch.utils.data.Subset(self.dataset_test, [0]),
            batch_size = 1,
            num_workers = 0,
            shuffle=False
        )
        for i, o in loader_sample:
            return (i, o)
        
    def get_train_test_loader(self, batchSize):
        loader_train = DataLoader(
            self.dataset_train,
            batch_size = batchSize,
            num_workers = self.num_workers,
            shuffle=True
        )
        loader_test = DataLoader(
            self.dataset_test,
            batch_size = batchSize,
            num_workers = self.num_workers,
            shuffle=True
        )
        return (loader_train, loader_test)
    
class MyTargetTransform():

    def __init__ (self, one_hot_target):
        self.one_hot_target = one_hot_target

    def __call__(self, x):
        t = torch.zeros(10)
        t[x] = 1
        if self.one_hot_target:
            return t
        else:
            return x