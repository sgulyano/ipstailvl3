import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import bs4

from sklearn import tree
import numpy as np
import io

def get_node_edge(clf, fn, cn, 
    px_init = 10,
    py_init = 10, 
    x_step = 5,
    y_step = 3,
    DEBUG=False):
    

    dot_text = tree.export_graphviz(clf, out_file=None, 
                        feature_names=fn,  
                        class_names=cn,
                        filled=True,
                        special_characters=True)
    
    G = nx.Graph(read_dot(io.StringIO(dot_text)))

    edge_x = []
    edge_y = []

    def append_edge(p1x, p2x, p1y, p2y):
        edge_x.append(p1x)
        edge_x.append(p2x)
        edge_x.append(None)
        edge_y.append(p1y)
        edge_y.append(p2y)
        edge_y.append(None)

    node_x = []
    node_y = []
    node_text = []
    node_col = []
    
    node_list = [('0',0,px_init,py_init)]

    while node_list:
        root, dep, px, py = node_list.pop()
        if DEBUG:
            print(f'root: {root}, depth={dep}, node_list={node_list}')
        adj_edge = [e for e in G.edges() if e[0]==root]

        node_x.append(px)
        node_y.append(py)
        node_text.append(bs4.BeautifulSoup(G.nodes[root]['label'][1:-1], features = "lxml").get_text('<br>'))
        node_col.append(G.nodes[root]['fillcolor'][1:-1])
        
        if len(adj_edge) == 2:
            sx = x_step / (2**dep)
            sy = -y_step * (dep + 1)
            
            node_list.append((adj_edge[0][1], dep+1, px-sx, py+sy))
            append_edge(px, px-sx, py, py+sy)
            
            node_list.append((adj_edge[1][1], dep+1, px+sx, py+sy))  
            append_edge(px, px+sx, py, py+sy)
        elif len(adj_edge) == 0:
            pass
        else:
            if DEBUG:
                print('ERROR', len(adj_edge))

    node = {'x':node_x, 'y':node_y, 'text':node_text, 'color':node_col}
    edge = {'x':edge_x, 'y':edge_y}


    anno = {'x':[px_init-x_step/2, px_init+x_step/2],
            'y':[py_init-y_step/2, py_init-y_step/2],
            'text':['True', 'False']}

    return node, edge, anno