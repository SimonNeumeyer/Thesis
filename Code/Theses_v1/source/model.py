from diffNN import *
import graph
import torch.nn as nn
import torch.nn.functional as F

class Model0(nn.Module):

    def __init__(self, settings):
        super(Model0, self).__init__()
        self.settings = settings
        features = self.settings["features"]
        classes = self.settings["classes"]
        self.conv_a = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
        self.linear_a = nn.Linear(1 * features, classes)

    def forward (self, x):
        x = F.relu(self.conv_a(x))
        x = torch.flatten(x, 1)
        x = self.linear_a(x)
        return x

class Model1(nn.Module):

    def __init__(self, settings):
        super(Model1, self).__init__()
        self.settings = settings
        self.linear_0 = nn.Linear(self.settings["features"], self.settings["graphs"]["features"])
        self.linear_1 = nn.Linear(self.settings["graphs"]["features"], self.settings["classes"])
        self.cells = nn.ModuleList()
        for i, cell in enumerate(self.settings["cells"]):
            graphs = graph.GraphGenerator(cell["numberNodes"]).get_random_subset(cell["numberGraphs"])
            #if self.settings["darts"]["sharedWeights"]:
            #    graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features_graphNN)
            #else:
            graphNNs = [GraphNN(graph, self.settings["graphs"]) for graph in graphs]
            diffNN = DiffNN(f"Cell_{i}", graphNNs, self.settings)
            self.cells.append(diffNN)

    def preprocess (self, x):
        return x.view(x.size()[0],-1)

    def forward (self, x):
        x = self.preprocess(x)
        x = F.relu(self.linear_0(x))
        for cell in self.cells:
            x = cell(x)
        x = self.linear_1(x)
        return x