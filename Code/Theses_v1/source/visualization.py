import matplotlib.pyplot as plt
import matplotlib.colors as colors
from util import *
import numpy
import torch
from tensorboardX import SummaryWriter
from networkx import draw_networkx

def visualize_flag(func):
    def wrapper_visualize_flag(self, *args, **kwargs):
        if self.visualize:
            func(self, *args, **kwargs)
    return wrapper_visualize_flag

def convert_tensor(tensor):
    return tensor.cpu()

class Visualization:
    
    def __init__(self, settings, settingsString):
        self.visualize = settings["visualize"]
        fileName = settings["fileName"]
        if self.visualize:
            self.writer = SummaryWriter(path_tensorboard() / fileName)
            self.title_training_loss = "Training loss"
            self.title_evaluation_loss = "Evaluation loss"
            self.title_accuracy = "Accuracy"
            self.diffNN_registry = []
            self.running_loss = 0
            self._write_settings(settingsString)
        
    def register_diffNN(self, diffNN):
        self.diffNN_registry.append(diffNN)
        
    def _visualize_graph(self, networkx_graph):
        draw_networkx(networkx_graph)
        return plt.gcf()

    def _write_settings(self, settingsString):
        self.writer.add_text("Settings", settingsString)

    @visualize_flag
    def plot_model(self, model, input=None):
        self.writer.add_graph(model, input)
        
    @visualize_flag
    def plot_graphs(self, graphs):
        for i, g in enumerate(graphs):
            fig = self._visualize_graph(g)
            self.writer.add_figure(figure=fig, tag="_".join(["Graphs"]), global_step=i)
        
    @visualize_flag
    def training_loss(self, epoch, counter, loss):
        """ 
        Args:
            loss: batch loss
        
        Collects loss information for learning rate visualization.
        """
        self.running_loss = ((counter - 1) * self.running_loss + loss) / counter

    @visualize_flag
    def evaluation_loss(self, epoch, evaluation_loss, accuracy):
        """ 
        Args:
            alphas: dictionary with 'title' - 'weights' as key - value structure
            optimizationSettings: optimizationSettings object
        
        Visualizes losses and given alphas to TensorBoard.
        """
        #Training loss
        self.writer.add_scalar(tag=self.title_training_loss, 
                               scalar_value=self.running_loss, 
                               global_step=epoch)
        self.running_loss = 0
        #Evaluation loss
        self.writer.add_scalar(tag=self.title_evaluation_loss, 
                               scalar_value=evaluation_loss, 
                               global_step=epoch)
        #Accuracy
        self.writer.add_scalar(tag=self.title_accuracy,
                               scalar_value=accuracy,
                               global_step=epoch)
        #alphas
        for diffNN in self.diffNN_registry:
            if diffNN.alpha_gradient_active():
                self.writer.add_figure(figure=self._get_alpha_plot(alpha=diffNN.get_alphas()),
                                        tag="_".join([diffNN.get_name(), Constants.ALPHA]),
                                        global_step=epoch)
            
    def close(self):
        self.writer.flush()
        self.writer.close()
        
    def _get_alpha_plot(self, alpha):
        return self._get_matrix_plot(alpha.cpu().unsqueeze(0))
    
    def _get_matrix_plot(self, matrix):
        if matrix.device != 'cpu':
            matrix = matrix.cpu()
        x = numpy.arange(-0.5, matrix.shape[1], 1)
        y = numpy.arange(-0.5, matrix.shape[0], 1)
        fig, ax = plt.subplots()
        #c = ax.pcolormesh(x, y, matrix, cmap='RdBu', norm=colors.Normalize())
        c = ax.pcolormesh(x, y, matrix, cmap='RdBu')
        fig.colorbar(c, ax=ax)
        return fig

    #@visualize_flag
    #def plot_weight_matrix(self, matrix, tag="default_tag"):
    #    self.writer.add_figure(figure=self._get_matrix_plot(matrix), tag=tag, global_step=TODO
