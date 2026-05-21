# ---------- imports ----------

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------- global variables + data load ----------

NUM_INDIVIDUALS = 543

# load data
data = np.load('WhaleSharkData/WhaleSharkID.npz')
meta_data = np.load('WhaleSharkData/WhaleSharkMeta.npz', allow_pickle=True)
gt = np.load('WhaleSharkData/WhaleSharkGT.npz', allow_pickle=True)

# (7693, 7693), cosine similarity
sim = data['similarity']
# ground truth similarity matrix ([i][j] is == 1 if i and j are of the same individual, else 0)
gt_sim = gt['gt_similarity']
# classes
classes = meta_data['classes']

# ---------- tools ----------

def trim(M, y, individuals):
    # trims the similarity matrix to include only images for the first n unique individuals
    mask = np.array([c in individuals for c in y])
    return M[mask][:, mask], y[mask], 

def nx_plot_graph(M, y):
    # makes a nice drawing using networkx for relatively small graphs given adjancency matrix M and class vector y
    np.fill_diagonal(M, 0)
    G = nx.from_numpy_array(M)
    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=y, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')

def greedy_clique_collapse_plot(M, y, min_size=3):
    # given an adjaceny matrix M and a class vector y
    # construct a graph G from M, and iteratively collapse the largest clique of size at least min_size into a single vertex
    # keeps track of any edges from non-clique vertices into clique vertices, and once there are no more cliques to
    # collapse adds those edges back to the graph (pointing to the correct new node)
    # the biggest limitation here is that plotting is not feasible for large graphs
    # so this method is only used when we restrict to a few individuals (<50)
    np.fill_diagonal(M, 0)
    G = nx.from_numpy_array(M)

    edges_to_add_back = []

    while True:

        largest_clique = list(max(nx.find_cliques(G), key=len))
        
        if len(largest_clique) < min_size:
            break

        else:
            # print(largest_clique)
            neighbors_of_clique = set()
            for node in largest_clique:
                neighbors_of_clique.update(G.neighbors(node))
                
            external_neighbors = [neighbor for neighbor in neighbors_of_clique if neighbor not in largest_clique]
            external_neighbors = set(external_neighbors)

            G.remove_nodes_from(largest_clique)
            collapsed_node = min(largest_clique)
            G.add_node(collapsed_node)

            edges_to_add_back += [(collapsed_node, neighbor) for neighbor in external_neighbors]

    G.add_edges_from(edges_to_add_back)

    colors = [y[node] for node in G.nodes()]

    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')

    return G

# ---------- main ----------

if __name__ == '__main__':

    # # ---------- randomly choose n individuals and trim sim & classes for those individuals ----------
    n = 3
    threshold = 0.5
    
    # randomly choose some individuals
    individuals = np.random.choice(range(1, NUM_INDIVIDUALS + 1), n)

    sim_trimmed, classes_trimmed = trim(sim, classes, individuals)
    gt_sim_trimmed, _ = trim(gt_sim, classes, individuals)

    # ---------- plot GT graph, thresholded sim graph, and collapsed thresholded sim graph ----------

    nx_plot_graph(gt_sim_trimmed, classes_trimmed)
    plt.title('Ground Truth Similarity Graph')

    nx_plot_graph(sim_trimmed > threshold, classes_trimmed)
    plt.title('Thresholded Similarity Graph')

    G = greedy_clique_collapse_plot(sim_trimmed > threshold, classes_trimmed)    
    plt.title('Collapsed Thresholded Similarity Graph')
    plt.show()