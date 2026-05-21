# ---------- imports ----------

import numpy as np
import matplotlib.pyplot as plt
from src.methods import nested_is
# from src.nis_modified import modified_nested_is
from src.nis_no_replacement import no_resampling_nis

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

def MaxGumbel(q, k):
    # q is list of (weight, value)
    weights = np.array([item[0] for item in q])
    values = np.array([item[1] for item in q])

    nonzero_idx = np.where(weights > 0)[0]
    if len(nonzero_idx) == 0:
        return np.array([]), np.array([])
    
    active_weights = weights[nonzero_idx]
    active_values = values[nonzero_idx]

    u = -np.log(np.random.random(len(active_weights))) / active_weights
    
    k = min(k, len(active_weights))
    indices_sampled = np.argpartition(u, k)[:k]
    ordered_indices_sampled = indices_sampled[np.argsort(u[indices_sampled])]
    
    final_values = active_values[ordered_indices_sampled]
    
    return final_values

# ---------- main ----------
if __name__ == '__main__':

    runs = 1000

    neighbor_counts = np.sum(gt_sim, axis=1)
    average_true_neighbors = int(np.mean(neighbor_counts))
    median_true_neighbors = int(np.median(neighbor_counts))
    max_true_neighbors = int(np.max(neighbor_counts))

    for threshold in [.9, .95]:

        for num_nghbrs in [average_true_neighbors, median_true_neighbors, max_true_neighbors]:

            avg_ratio = 0
            avg_new_dist_ratio = 0

            thresholded_sim = sim > threshold

            for i in range(runs):

                # choose random vertex
                vertex = np.random.choice(len(sim[0]))
                
                # sample num_nghbrs proportional to the similarity
                q = [(sim[vertex][x], x) for x in range(len(sim[vertex]))]
                sampled_neighbors = MaxGumbel(q, num_nghbrs)

                ratio = sum(gt_sim[vertex][sampled_neighbors]) / sum(gt_sim[vertex])
                avg_ratio += ratio

                # proportional to the average similarity to the thresholded neighbors of vertex
                high_confidence_neighbors = np.where(thresholded_sim[vertex] == 1)[0]

                # 1) Simple Averaging
                # q = [(sum(sim[x][high_confidence_neighbors])/len(high_confidence_neighbors), x) for x in range(len(sim[vertex]))]
                # sampled_neighbors = MaxGumbel(q, num_nghbrs)

                # 2) Multiplicative Gating
                if len(high_confidence_neighbors) > 1:
                    q = [(sim[vertex][x] * (sum(sim[x][high_confidence_neighbors]) - sim[vertex][x]) / (len(high_confidence_neighbors) - 1), x) for x in range(len(sim[vertex]))]
                else:
                    q = [(sim[vertex][x], x) for x in range(len(sim[vertex]))]

                sampled_neighbors = MaxGumbel(q, num_nghbrs)

                new_dist_ratio = sum(gt_sim[vertex][sampled_neighbors]) / sum(gt_sim[vertex])
                avg_new_dist_ratio += new_dist_ratio
            
            print(f'When threshold is {threshold}, and we sample {num_nghbrs} neighbors for a vertex:')
            print(f'default: neighbors found / true neighbors is {avg_ratio/runs}.')
            print(f'modified dist.: neighbors found / true neighbors {avg_new_dist_ratio/runs}.')
            print('')


# MOST RECENT RUN

# When threshold is 0.9, and we sample 40 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.05887054686668945.
# modified dist.: neighbors found / true neighbors 0.10734080596389815.

# When threshold is 0.9, and we sample 26 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.04052705178450948.
# modified dist.: neighbors found / true neighbors 0.06533561245131696.

# When threshold is 0.9, and we sample 154 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.20374146268163837.
# modified dist.: neighbors found / true neighbors 0.2791782804955742.

# When threshold is 0.95, and we sample 40 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.061677449671510336.
# modified dist.: neighbors found / true neighbors 0.08106470868750434.

# When threshold is 0.95, and we sample 26 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.04070714164454938.
# modified dist.: neighbors found / true neighbors 0.05124892435717542.

# When threshold is 0.95, and we sample 154 neighbors for a vertex:
# default: neighbors found / true neighbors is 0.21793980111590408.
# modified dist.: neighbors found / true neighbors 0.26548955596377655.