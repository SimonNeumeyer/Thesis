import matplotlib.pyplot as plt
import matplotlib.colors as colors
from util import *
import numpy
from tensorboardX import SummaryWriter
import networkx

        
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
