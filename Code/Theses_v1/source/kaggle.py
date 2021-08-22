
import torch 
import torchvision
from torch.utils.data import Dataset
import torch
from networkx import DiGraph
import torch.nn as nn
import numpy
import networkx
from collections import OrderedDict
#from myLogging import alpha_gradient_logging
from torch.optim import SGD
from pathlib import Path
import os
import matplotlib.colors as colors
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from numpy.random import default_rng
from torch.utils.data import DataLoader
import torch.nn.functional as functional
import datetime
import matplotlib.pyplot as plt

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
        self.batch_size = 4
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
        
    def get_train_test_loader(self):
        self.number_classes = 2
        self.train_test_ratio = 0.8
        loader_train = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(0, int(self.train_test_ratio * self.size))),
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        loader_test = DataLoader(
                self,
                sampler=SubsetRandomSampler(range(int(self.train_test_ratio * self.size), self.size)),
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        return (loader_train, loader_test)

class MyMNIST():
    def __init__(self):
        self.path = Constants.PATH_DATA / "MNIST"
        try:
            os.makedirs(self.path)
        except:
            pass
        self.number_classes = 10
        self.number_features = 28 * 28
        self.batch_size = 64
        self.num_workers = 4
        
    def get_train_test_loader(self):
        loader_train = DataLoader(
            torchvision.datasets.MNIST(
                self.path,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    MyTransformFromImage()]),
                target_transform=MyTargetTransform()),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=True
        )

        loader_test = DataLoader(
            torchvision.datasets.MNIST(
                self.path,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    MyTransformFromImage()]),
                target_transform=MyTargetTransform()),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=True
        )
        return (loader_train, loader_test)
    
class MyTransformFromImage():
    def __call__(self, x):
        return torch.squeeze(x.view(1,-1,1))
    
class MyTargetTransform():
    def __call__(self, x):
        t = torch.zeros(10)
        t[x] = 1
        return t


class DiffNN(nn.ModuleList):
    
    def __init__(self, modules, optimization_settings):
        super(DiffNN, self).__init__(modules)
        self.alphas = self.init_alphas(modules)
        self.set_alpha_update(False)
        self.name = "OnlyDiffNN" #TODO
        self.optimization_settings = optimization_settings
        
    def get_name(self):
        return self.name
    
    def set_optimization_settings(self, optimization_settings):
        self.optimization_settings = optimization_settings
        
    def set_alpha_update(self, alpha_update):
        """ Freeze or unfreeze weights respectively alphas """
        self.alpha_update = alpha_update
        if alpha_update:
            self.alphas.requires_grad = True
            self.set_weight_update(False)
        else:
            self.alphas.requires_grad = False
            self.set_weight_update(True)
            
    def get_alphas(self):
        return self.alphas.data.clone().detach()
            
    def get_weight_parameters(self):
        return [parameter[1] for parameter in self.named_parameters() if "alphas" not in parameter[0]]
            
    def set_weight_update(self, weight_update):
        for parameter in self.get_weight_parameters():
            if weight_update:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        
    def calculate_alphas(self, alpha_sampling):
        """ alphas are either sampled or smooth """
        alphas = nn.functional.softmax(self.alphas, dim=0)
        if alpha_sampling:
            alphas = nn.functional.gumbel_softmax(self.alphas, hard=True, tau=1)
        return alphas
        
    def init_alphas(self, modules):
        alphas = nn.Parameter(torch.ones(len(modules)), requires_grad=True)
        #alphas.register_hook(alpha_gradient_logging)
        return alphas
    
    def alpha_gradient_active(self):
        return self.alpha_update
        
    def reduce(self, alphas, tensors, reduce, normalize):
        if reduce == Constants.REDUCE_FUNC_SUM:
            #reduced = nn.functional.linear(alphas, torch.stack(tensors, dim=2))
            reduced = torch.matmul(alphas, torch.stack(tensors, dim=1))
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x):
        if self.optimization_settings.alpha_update and not self.alpha_update:
            self.set_alpha_update(True)
        if not self.optimization_settings.alpha_update and self.alpha_update:
            self.set_alpha_update(False)
        alphas = self.calculate_alphas(self.optimization_settings.alpha_sampling)
        return self.reduce(alphas, [module(x, self.optimization_settings) for module in self], reduce=self.optimization_settings.diffNN_reduce, normalize=self.optimization_settings.diffNN_normalize)
        
        
class GraphNN(nn.Module):
    
    def __init__(self, graph, width=None, shared_edgeNNs=None):
        super(GraphNN, self).__init__()
        self.graph = graph
        self.width = width
        self.edgeNN = "EdgeNN"
        self.initModel(shared_edgeNNs)
        
    @classmethod
    def generate_graphNNs_shared_weights(cls, graphs, width):
        graphNNs = []
        shared_edgeNNs = cls.generate_shared_weights(graphs[0], width)
        for g in graphs:
            graphNNs.append(GraphNN(g, shared_edgeNNs=shared_edgeNNs))
        return graphNNs
            
    @classmethod
    def generate_shared_weights(cls, graph, width):
        graph = GraphGenerator.get_maximal_phenotype(graph.number_nodes_without_input_output())
        return cls.create_edgeNNs(graph, width)
    
    @classmethod
    def create_edgeNNs(cls, graph, width):
        edgeNNs = {}
        for edge in graph.edges():
            if graph.input_output_edge(*edge):
                edgeNN = nn.Identity() 
            else:
                edgeNN = nn.Sequential(nn.Linear(width, width), nn.ReLU())
            edgeNNs[str(edge)] = edgeNN
        return edgeNNs
        
    def initModel(self, edgeNNs):
        if edgeNNs is None:
            edgeNNs = self.create_edgeNNs(self.graph, self.width)
        for edge in self.graph.edges():
            if self.graph.input_output_edge(*edge):
                edgeNN = nn.Identity() 
            else:
                edgeNN = edgeNNs[str(edge)]
            self.graph.set_edge_attribute(*edge, self.edgeNN, edgeNN) # self.edgeNN is string constant
        self.edgeNNs = nn.Sequential(OrderedDict(edgeNNs)) # register model
        
    def reduce(self, tensors, reduce, normalize):
        if reduce == Constants.REDUCE_FUNC_SUM:
            reduced = torch.sum(torch.stack(tensors), dim=0)
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x, optimization_settings):
        outputs = {self.graph.input_node : x}
        for v in self.graph.ordered_nodes(except_input_node = True):
            inputs = [self.graph.get_edge_attribute(p, v)[self.edgeNN](outputs[p]) for p in self.graph.get_predecessors(v)]
            outputs[v] = self.reduce(inputs, reduce=optimization_settings.graphNN_reduce, normalize=optimization_settings.graphNN_normalize)
        return outputs[self.graph.output_node]

    
class Graph():
    
    def __init__(self, vertices, edges):
        self.lib_graph = DiGraph()
        self.lib_graph.add_nodes_from(vertices)
        self.lib_graph.add_edges_from(edges)
        self.vertices_except_input_output = list(self.lib_graph.nodes)
        self.vertices_except_input_output.sort()
        self.input_node = "INPUT"
        self.output_node = "OUTPUT"
        self.append_input_output_node()
        
    def __str__(self):
        return str(self.lib_graph.edges)
    
    def append_input_output_node(self):
        self.lib_graph.add_edges_from([(self.input_node, v) for v in self.lib_graph.nodes() if not any(True for _ in self.lib_graph.predecessors(v))])
        self.lib_graph.add_edges_from([(v, self.output_node) for v in self.lib_graph.nodes() if not any(True for _ in self.lib_graph.successors(v))])
    
    def set_edge_attribute(self, v_from, v_to, key, value):
        assert all([v in self.lib_graph.nodes for v in [v_from, v_to]]), "Edge to manipulate not contained in graph"
        self.lib_graph[v_from][v_to][key] = value
        
    def get_edge_attribute(self, v_from, v_to):
        assert all([v in self.lib_graph.nodes for v in [v_from, v_to]]), "Edge to manipulate not contained in graph"
        return self.lib_graph[v_from][v_to]
    
    def get_predecessors(self, v_to):
        assert v_to in self.lib_graph.nodes(), "Node not contained in graph"
        return self.lib_graph.predecessors(v_to)
    
    def input_output_edge(self, v_from, v_to):
        return v_from == self.input_node or v_to == self.output_node
    
    def get_networkx(self):
        return self.lib_graph
    
    def edges(self):
        return self.lib_graph.edges
    
    def number_nodes_without_input_output(self):
        return len(self.lib_graph.nodes) - 2
    
    def ordered_nodes(self, except_input_node = True):
        if not except_input_node:
            return [self.input_node] + self.vertices_except_input_output + [self.output_node]
        else:
            return self.vertices_except_input_output + [self.output_node]
        
class GraphGenerator():
    
    def __init__(self, number_nodes):
        assert number_nodes > 1, "Number of nodes needs to be greater than 1"
        self.rand = default_rng(0)
        self.number_nodes = number_nodes
        self.graphs = self.init_graphs()
        self.clean_isomorphism()
        
    @classmethod
    def get_maximal_phenotype(cls, number_nodes):
        graph = {}
        for i in range(1, number_nodes):
            graph[number_nodes - i] = [int(c) for c in f"{2**i-1:0{number_nodes}b}"]
        return cls.contain_graph_from_technical(number_nodes, graph)
    
    @classmethod
    def contain_graph_from_technical(cls, number_nodes, technical_graph):
        edges = []
        vertices = numpy.array(range(1, number_nodes + 1))
        for v in technical_graph:
            edges.extend([(v, s) for s in vertices[numpy.array(technical_graph[v], dtype=bool)]])
        return Graph(vertices, edges)
        
    def init_graphs(self):
        graphs = [{}]
        for i in range(1, self.number_nodes):
            one_node_smaller_graphs = graphs
            graphs = []
            for one_node_smaller_graph in one_node_smaller_graphs:
                for j in range(2 ** i):
                    graph = one_node_smaller_graph.copy()
                    graph[self.number_nodes - i] = [int(c) for c in f"{j:0{self.number_nodes}b}"]
                    graphs.append(graph)
        return [self.contain_graph_from_technical(self.number_nodes, g) for g in graphs]
    
    def clean_isomorphism(self):
        cleaned = []
        for g in self.graphs:
            if not any([networkx.is_isomorphic(g.get_networkx(), h.get_networkx()) for h in cleaned]):
                cleaned.append(g)
        #print(f"clean iso: before {len(self.graphs)}, after {len(cleaned)}")
        self.graphs = cleaned

    
    def number_graphs(self):
        return len(self.graphs)
    
    def get_random_subset(self, number_samples):
        indices = self.rand.choice(self.number_graphs(), number_samples, replace=False)
        return [self.get(index) for index in indices]
    
    def get(self, index):
        return self.graphs[index]
        

def alpha_gradient_logging(gradient):
    print(gradient)


if False:
    print(hash((2,3)))
    print(hash((3,3)))
    print(hash((2,3)))

#a = torch.ones(5)
# a = torch.tensor([-5,-1,2,3,4,5], dtype=torch.float32)
# a.requires_grad = True
# a.register_hook(alpha_gradient_logging)
# for i in range(10):
#     gumbel = functional.gumbel_softmax(logits = a, hard=True, tau=10)
#     for p in [gumbel, a]:
#         if p.grad is not None:
#             p.grad.detach_()
#             p.grad.zero_()
#     gumbel.max().backward()
# print("done")
# g = graph.GraphGenerator(3)
# graph = g.graphs[3]
# graphNN = diffNN.GraphNN(graph, 10)
# i = torch.ones(10)
# print(graphNN(i))


class Constants():
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    PATH_DATA = Path("/kaggle/working/data")
    
def path_tensorboard():
    return Path(f"/kaggle/working/runs/{datetime.datetime.now().strftime('%b%d%H%M%S')}")

def copy_tensor(tensor):
    return tensor.data.clone()


        
def visualize_flag(func):
    def wrapper_visualize_flag(self, *args, **kwargs):
        if self.visualize:
            func(self, *args, **kwargs)
    return wrapper_visualize_flag


class Visualization:
    
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.writer = SummaryWriter(path_tensorboard())
        self.title_training_loss = "Training loss"
        self.title_evaluation_loss = "Evaluation loss"
        self.diffNN_registry = []
        self.batch_count = 0
        self.epoch_count = 0
        self.running_loss = 0
        
    def register_diffNN(self, diffNN):
        self.diffNN_registry.append(diffNN)
        
    def _visualize_graph(self, networkx_graph):
        networkx.draw_networkx(networkx_graph)
        return plt.gcf()
        
    @visualize_flag
    def plot_graphs(self, graphs):
        for i, g in enumerate(graphs):
            fig = self._visualize_graph(g)
            self.writer.add_figure(figure=fig, tag="_".join(["Graphs"]), global_step=i)
        
    @visualize_flag
    def batch(self, loss):
        """ 
        Args:
            loss: batch loss
        
        Collects loss information for learning rate visualization.
        """
        self.batch_count += 1
        self.running_loss = ((self.batch_count - 1) * self.running_loss + loss) / self.batch_count
                
    @visualize_flag
    def epoch(self, evaluation_loss):
        """ 
        Args:
            alphas: dictionary with 'title' - 'weights' as key - value structure
            optimizationSettings: optimizationSettings object
        
        Visualizes losses and given alphas to TensorBoard.
        """
        self.epoch_count += 1
        self.batch_count = 0
        #Training loss
        self.writer.add_scalar(tag=self.title_training_loss, 
                               scalar_value=self.running_loss, 
                               global_step=self.epoch_count)
        self.running_loss = 0
        #Evaluation loss
        self.writer.add_scalar(tag=self.title_evaluation_loss, 
                               scalar_value=evaluation_loss, 
                               global_step=self.epoch_count)
        #alphas
        for diffNN in self.diffNN_registry:
            if diffNN.alpha_gradient_active():
                self.writer.add_figure(figure=self.get_alpha_plot(alpha=diffNN.get_alphas()),
                                        tag="_".join([diffNN.get_name(), Constants.ALPHA]),
                                        global_step=self.epoch_count)
            
    def close(self):
        self.writer.flush()
        self.writer.close()
        
    def get_alpha_plot(self, alpha):
        return self.get_matrix_plot(alpha.unsqueeze(0).numpy())
    
    def get_matrix_plot(self, matrix):
        x = numpy.arange(-0.5, matrix.shape[1], 1)
        y = numpy.arange(-0.5, matrix.shape[0], 1)
        fig, ax = plt.subplots()
        #c = ax.pcolormesh(x, y, matrix, cmap='RdBu', norm=colors.Normalize())
        c = ax.pcolormesh(x, y, matrix, cmap='RdBu')
        fig.colorbar(c, ax=ax)
        return fig
    
    @visualize_flag
    def plot_weight_matrix(self, matrix, tag="default_tag"):
        self.writer.add_figure(figure=self.get_matrix_plot(matrix), tag=tag, global_step=self.epoch_count)



class Trainer():
    
    def __init__(self, number_nodes=3, number_graphs=3, visualize=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initiate_data()
        self.visualization = Visualization(visualize)
        self.optimization_settings = OptimizationSettings()
        graphs = graph.GraphGenerator(number_nodes).get_random_subset(number_graphs)
        if self.optimization_settings.shared_weights:
            graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features)
        else:
            graphNNs = [GraphNN(graph, self.number_features) for graph in graphs]
        diffNN = DiffNN(graphNNs, self.optimization_settings)
        self.visualization.plot_graphs([g.get_networkx() for g in graphs])
        self.visualization.register_diffNN(diffNN)
        self.model = nn.Sequential(diffNN, nn.Linear(self.number_features, self.number_classes), nn.Softmax(dim=-1))
        self.model.to(self.device)
        
        self.loss_function, self.optimizer = self.backward_stuff(self.model)
        
    def initiate_data(self):
        #dataset = MyDataset(overwrite=True)
        dataset = MyMNIST()
        self.loader_train, self.loader_test = dataset.get_train_test_loader()
        self.number_classes = dataset.number_classes
        self.number_features = dataset.number_features
        
        
    def backward_stuff(self, model):
        return (nn.CrossEntropyLoss(), SGD(model.parameters(), lr=1e-1, momentum=0.8))
    
    def run(self):
        for i in range(5):
            print(f"epoch {i+1}")
            self.update_optimization_settings(alpha_update=False)
            for j in range(1):
                self.train_epoch()
                self.evaluate()
            self.update_optimization_settings(alpha_update=True)
            for j in range(1):
                self.train_epoch()
                self.evaluate()
    
    def train_epoch(self):
        self.model.train()
        for i, o in self.loader_train:
            i, o = i.to(self.device), o.to(self.device)
            loss = self.loss(i, o)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visualization.batch(copy_tensor(loss))
        self.visualization.epoch(copy_tensor(self.evaluate()))
        for m in self.model:
            if isinstance(m, nn.Linear):
                self.visualization.plot_weight_matrix(copy_tensor(m.weight), "Last linear")
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            batch_count = 0
            for i, o in self.loader_test:
                i, o = i.to(self.device), o.to(self.device)
                batch_count +=1
                loss = ((batch_count - 1) * loss + self.loss(i, o)) / batch_count
        self.model.train()
        return loss
    
    def update_optimization_settings(self, **kwargs):
        self.optimization_settings.update(**kwargs)
        for m in self.model:
            if isinstance(m, DiffNN):
                m.set_optimization_settings(self.optimization_settings)
                
    def loss(self, i, o):
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            _, o = o.max(dim=-1) #get indices from one-hot encoding
        return self.loss_function(self.model(i), o)
        
    def close(self):
        self.visualization.close()
        
        
class OptimizationSettings():
    
    def __init__(self, **kwargs):
        self._init_default()
        self.update(**kwargs)
        
    def _init_default(self):
        self.alpha_update = True
        self.alpha_sampling = False
        self.diffNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_normalize = True
        self.diffNN_normalize = False
        self.shared_weights = False
        
    def update(self, **kwargs):
        for key in kwargs:
            assert key in self.__dict__, f"Setting '{key}' not available"
            self.__dict__[key] = kwargs[key]
    
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()