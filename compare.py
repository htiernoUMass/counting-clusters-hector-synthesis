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

# ---------- NIS ----------
def NIS(sim, gt_sim, runs, max_N, ratio=1, lw=1.5, fs=9):

    N = [1] + list(range(1,max_N,3))

    f_hat_nis_all = []
    f_hat_m_nis_all = []
    n_hat = None

    for n in N: 
        f_hat_nis_ = []
        f_hat_m_nis_ = []
        Nv = max(1, round(n/np.sqrt(ratio)))
        Nn = max(1, round(ratio/np.sqrt(ratio)*n))

        for run in range(runs):

            print(f'run {run+1} for n={n}     ', end='\r')

            f_hat_nis, _, n_hat = nested_is(gt_sim, sim, Nv, Nn, ci=True, n_hat=n_hat)
            f_hat_m_nis, _ = no_resampling_nis(gt_sim, sim, Nv, Nn, n_hat=n_hat)

            f_hat_m_nis_.append(f_hat_m_nis)
            f_hat_nis_.append(f_hat_nis)

        f_hat_m_nis_all.append(np.mean(f_hat_m_nis_))
        f_hat_nis_all.append(np.mean(f_hat_nis_))

    # Plot results
    tn = len(sim[0])
    x = np.array([n**2 for n in N], dtype=np.float64)
    x /= ((tn*(tn-1))/2)
    x*=100


    fig, ax = plt.subplots(1,1, figsize=(2.7,2.3))
    plt.plot(x, f_hat_m_nis_all, color='maroon', linestyle='--', linewidth=lw, label='Nested-m-IS') 
    plt.plot(x, f_hat_nis_all, color='blue', linestyle='--', linewidth=lw, label='Nested-IS') 
    plt.plot(x, NUM_INDIVIDUALS*np.ones(len(x)))
    plt.ylabel('Estimated count', fontsize=fs)
    plt.xlabel('# sampled pairs / # total edges', fontsize=fs)
    plt.xticks([0, 0.016, 0.032], [0, '0.016%', '0.032%'], rotation=0, fontsize=fs)
    plt.xlim(-0.001, 0.2)
    plt.grid(True, color='gray', linestyle=':', linewidth=0.5)
    plt.legend(fontsize=fs)
    plt.title('Comparing NIS and NIS with no Replacement')
    fig.tight_layout()
    plt.show()

# ---------- main ----------
if __name__ == '__main__':

    NIS(sim, gt_sim, runs = 5, max_N = 102, ratio=1)