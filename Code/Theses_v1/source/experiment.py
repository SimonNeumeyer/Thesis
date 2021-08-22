import torch
import torch.nn as nn
from settings import Settings

class Experiment():

    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def run(self):
        input = torch.rand(1,1,32,32)
        output = self.model(input)
        print(output.size())
        print("done")

class A():
    def __init__ (self):
        self.z = {"a": "old"}
        self.b = B(self.z)

    def test (self):
        print(self.z)
        self.b.test()

class B():
    def __init__ (self, dict):
        self.z = dict

    def test (self):
        print(self.z)

if __name__ == "__main__":
    settings = Settings()
    print("done init")
    settings = Settings(**{"model": {"classes": 2, "features": 2}})
    print("done complex init")
    settings.update(**{"model": {"classes": 2, "features": 2}})
    print("done update")