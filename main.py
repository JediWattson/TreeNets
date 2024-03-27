import math
import tiktoken
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch import argmax, cat, empty, zeros, matmul, tensor
from graph import Graph
from numpy.random import normal

window_size = 16
batch_size = 24
scaling_factor = math.sqrt(window_size * batch_size)
limit = math.sqrt(2 / float(batch_size + window_size))
loss_fn = CrossEntropyLoss()

def make_embedding_matrix(size):
    return tensor(normal(0.0, limit, size=size))

def make_graphs(tokens):
    k = Graph()
    v = Graph()
    q = Graph()

    pos = 0
    last_token = None
    
    for token in tokens:
        q.add_node(pos, token)
        if pos < len(tokens) - 1:
            emb_v = make_embedding_matrix(window_size)
            q.nodes[pos].add_edge(pos+1, emb_v)
        pos = pos + 1
    
        if not k.has_node(token):
            v.add_node(token, make_embedding_matrix(window_size))
            k.add_node(token, token)        
        if last_token is not None:
            emb_v = make_embedding_matrix(window_size)
            k.nodes[last_token].add_edge(token, emb_v)
        last_token = token
    return q, k, v


def traverse(q, start, end = None):
    hit = {}
    key = start
    while True:
        node = q.nodes[key]
        print(key, node.val, node.edges)
        keys = node.edges.keys()
        if key == end or len(keys) == 0 or (end is None and key in hit):
            break
        hit[key] = True
        key = list(keys)[0]
    
def infer_attention(q, k, v, start, end):
    pos = start
    count = 0
    while True:
        q_node = q.nodes[pos]
        k_node = k.nodes[q_node.val]
        v_node = v.nodes[q_node.val]
        
        k_edges = list(k_node.edges.items()) 
        q_weights = q_node.edges[pos+1]
        v_weights = v_node.val

        edges_len = len(k_edges)
        attention = empty(edges_len)
        optimizer = AdamW([q_weights, v_weights], lr=0.001)

        target_idx = -1
        next_val = q.nodes[pos+1].val
        attention = []
  
        for i, (key, k_weights) in enumerate(k_edges):
            if key == next_val:
                target_idx = i
            combined_weights = cat((q_weights, k_weights, v_weights), dim=-1)      
            score = matmul(combined_weights, combined_weights.transpose(-1, -1))
            attention.append(score)
        
        attention_tensor = softmax(tensor(attention, requires_grad=True), dim=-1)    
        one_hot_v = zeros(edges_len)
        one_hot_v[target_idx] = 1 
        lost = loss_fn(attention_tensor, one_hot_v)

        optimizer.zero_grad()
        lost.backward()
        optimizer.step()
        
        if argmax(attention_tensor) == target_idx:
            count = count + 1

        if pos == end:
            break
        pos = pos + 1

    return count / (end - start)

def init():
    print("=== generating tokens ===")
    shakespeare = open('shakespeare.txt', 'r').read()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(shakespeare[:9000])

    print("=== init graphs ===")
    q, k, v = make_graphs(tokens) 

    print("=== calculating attention ===")
    for epoch in range(200):
        hits = 0.0
        for i in range(batch_size):
            hits = hits + infer_attention(q, k, v, i, i+window_size)
        print("HITS %: ", hits/batch_size)
if __name__ == '__main__':
    init()
