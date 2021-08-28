from diffNN import *
import graph
import torch.nn as nn
import torch.nn.functional as F
import math

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

class MLP(nn.Module): # Multi Layer Perceptron

    def __init__(self, settings, diffNN_callback):
        super(MLP, self).__init__()
        self.settings = settings
        self.features = self.settings["features"]
        self.classes = self.settings["classes"]
        self.layer_width = self.settings["graphs"]["mlp"]["layerWidth"]
        self.shared_weights = self.settings["darts"]["sharedWeights"]
        self.linear_0 = nn.Linear(self.features, self.layer_width)
        self.linear_1 = nn.Linear(self.layer_width, self.classes)
        self.cells = nn.ModuleList()
        for i, cell in enumerate(self.settings["cells"]):
            self.graphs = graph.GraphGenerator(cell["numberNodes"]).get_random_subset(cell["numberGraphs"])
            if self.shared_weights:
                graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features_graphNN)
            else:
                graphNNs = [GraphNN(graph, self.settings["graphs"]) for graph in self.graphs]
            diffNN = DiffNN(f"Cell_{i}", graphNNs, self.settings)
            self.cells.append(diffNN)
            diffNN_callback(diffNN)

    def get_graphs (self):
        return [g.get_networkx() for g in self.graphs]

    def preprocess (self, x):
        return x.view(x.size()[0],-1)

    def set_alpha_update (self, alphaUpdate):
        for cell in self.cells:
            cell.set_alpha_update(alphaUpdate)

    def forward (self, x):
        x = self.preprocess(x)
        x = F.relu(self.linear_0(x))
        for cell in self.cells:
            x = cell(x)
        x = self.linear_1(x)
        return x

class ConvolutionalModel(nn.Module):

    def __init__(self, settings, diffNN_callback):
        super(ConvolutionalModel, self).__init__()
        self.settings = settings
        self.features = self.settings["features"]
        self.classes = self.settings["classes"]
        self.number_channels = self.settings["graphs"]["convolution"]["numberChannels"]
        self.shared_weights = self.settins["darts"]["sharedWeights"]
        self.cells = nn.ModuleList()
        self.poolings = nn.ModuleList()
        for i, cell in enumerate(self.settings["cells"]):
            self.graphs = graph.GraphGenerator(cell["numberNodes"]).get_random_subset(cell["numberGraphs"])
            if self.settings["darts"]["sharedWeights"]:
                graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features_graphNN)
            else:
                graphNNs = [GraphNN(graph, self.settings["graphs"]) for graph in self.graphs]
            diffNN = DiffNN(f"Cell_{i}", graphNNs, self.settings)
            self.cells.append(diffNN)
            diffNN_callback(diffNN)

            #pooling layer
            self.poolings.append(pool)

        self.linear_1 = nn.Linear()

    def get_graphs (self):
        return [g.get_networkx() for g in self.graphs]

    def set_alpha_update (self, alphaUpdate):
        for cell in self.cells:
            cell.set_alpha_update(alphaUpdate)

    def check_hyperparameters (self, final_channel_count, number_cells, number_features):
        assert any([math.isclose(math.log(i, number_features / 2), number_cells) for i in range(2,5)])

    def forward (self, x):
        for index, cell in enumerate(self.cells):
            x = cell(x)
            if index < len(self.cells) - 1:
                x = self.poolings[index](x)
        x = x.view(x.size()[0],-1)
        # TODO: wie hört CNN auf??k
        x = self.linear_1(x)
        return x