class Node:
    def __init__(self, val):
        self.val = val
        self.edges = {}
    def add_edge(self, key, emdbeddings = []):
        self.edges[key] = emdbeddings

class Graph:
    def __init__(self):
        self.nodes = {}
    def add_node(self, key, val):
        self.nodes[key] = Node(val)
    def has_node(self, key):
        return key in self.nodes
