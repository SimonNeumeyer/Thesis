import numpy


class GraphManager():
    
    def __init__(self, number_nodes):
        assert number_nodes > 1, "Number of nodes needs to be greater than 1"
        self.number_nodes = number_nodes
        self.graphs = self.init_graphs()
        
    def init_graphs(self):
        graphs = [{}]
        for i in range(1, self.number_nodes):
            one_node_smaller_graphs = graphs
            graphs = []
            for one_node_smaller_graph in one_node_smaller_graphs:
                for j in range(2 ** i):
                    graph = one_node_smaller_graph.copy()
                    graph[self.number_nodes - i] = j
                    graphs.append(graph)
        return [g for g in [self.append_first_and_last_node(Graph(graph)) for graph in graphs] if g]
    
    def append_first_and_last_node(self, graph):
        #TODO
        if numpy.array([numpy.array([graph.get_edges(node) for node in range(1, self.number_nodes)]).any()]).all():
            return graph
        else:
            return None
        
    def random(self, number_graphs):
        #TODO
        assert number_graphs <= len(self.graphs)
        return self.graphs[:number_graphs]
    
    def getGraph(self, index):
        return self.graphs[index]

class Graph():
    
    def __init__(self, edges):
        self.number_nodes = len(edges) + 1
        self.edges = edges
        
    def __str__(self):
        return str(self.edges)
        
    def get_edges(self, index):
        """ 1-indexed """
        assert 1 <= index and index < self.number_nodes
        return numpy.array([float(c) for c in f"{self.edges[index]:0{self.number_nodes - 1}b}"])
    
if __name__ == "__main__":
    number_nodes = 3
    m = GraphManager(number_nodes)
    print(m.graphs)
    for i in range(2 ** number_nodes):
        print(m.getGraph(i))
        for j in range(1, number_nodes):
            print(f"Edges {j}: {m.getGraph(i).get_edges(j)}")