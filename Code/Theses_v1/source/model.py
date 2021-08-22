from diffNN import *
import torch.nn as nn
import torch.nn.functional as F

class Model0(nn.Sequential):

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

class Model1(nn.Sequential):

    def __init__(self, settings):
        super(Model1, self).__init__()
        self.settings = settings
        graphs = graph.GraphGenerator(number_nodes).get_random_subset(number_graphs)
        if self.optimizationSettings.shared_weights:
            graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features_graphNN)
        else:
            graphNNs = [GraphNN(graph, self.number_features_graphNN) for graph in graphs]
        diffNN = DiffNN(graphNNs, self.optimization_settings)
        self.add_module("FirstLinear", nn.Linear(self.number_features_in, self.number_features_graphNN))
        self.add_module("DiffNN", diffNN)
        self.add_module("LastLinear", nn.Linear(self.number_features_graphNN, self.number_classes))

        #self.visualization.plot_graphs([g.get_networkx() for g in graphs])
        #self.visualization.register_diffNN(diffNN)

    def preprocess (self, x):
        return x.view(x.size()[0],-1)


if __name__ == "__main__":
    print(Model1().a())