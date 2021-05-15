import numpy
from networkx import DiGraph
    
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
    
    def edges(self):
        return self.lib_graph.edges
    
    def ordered_nodes(self, except_input_node = True):
        if not except_input_node:
            return [self.input_node] + self.vertices_except_input_output + [self.output_node]
        else:
            return self.vertices_except_input_output + [self.output_node]
        
class GraphGenerator():
    
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
                    graph[self.number_nodes - i] = [int(c) for c in f"{j:0{self.number_nodes}b}"]
                    graphs.append(graph)
        return [self.contain_graph_from_technical(g) for g in graphs]
    
    def contain_graph_from_technical(self, technical_graph):
        edges = []
        vertices = numpy.array(range(1, self.number_nodes + 1))
        for v in technical_graph:
            edges.extend([(v, s) for s in vertices[numpy.array(technical_graph[v], dtype=bool)]])
        return Graph(vertices, edges)
    
    def number_graphs(self):
        return len(self.graphs)
    
    def get(self, index):
        return self.graphs[index]

if __name__ == "__main__":
    number_nodes = 3
    m = GraphGenerator(number_nodes)
    for i in range(2 ** number_nodes):
        print(m.get(i))