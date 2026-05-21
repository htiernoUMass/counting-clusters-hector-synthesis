import os
import sys
import time
import argparse
import numpy as np
from os.path import join

sys.path.insert(0, 'src')
from methods import nested_is, nested_mc, calculate_similarity
from utils_ import load_features, plot_results

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Estimate population size in a dataset')
    parser.add_argument('--feats_dir', type=str, default='features', help='for features')
    parser.add_argument('--dataset', type=str, default='MacaqueFaces', help='for features')
    parser.add_argument('--model', type=str, default='L-384', help='for features')
    parser.add_argument('--pretraining', type=str, default='megad', help='for features')
    parser.add_argument('--metric', type=str, default='cosine', help='for similarity')
    parser.add_argument('--ratio', type=int, default=1, help='Nn/Nv ratio (use 7 for GiraffeZebraID)')
    parser.add_argument('--runs', type=int, default=10, help='number of runs to plot results')
    parser.add_argument('--seed', type=int, default=0, help='seed for features')
    parser.add_argument('--verbose', action='store_true', help='print intermediate progress')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    ti = time.time()
    print('[loading %s-%s-%s features...]'%(args.dataset, args.model, args.pretraining), end=" ")
    features, meta = load_features(args)
    print('done %s [%.1fs]'%(str(features.shape), time.time() - ti))

    a = [{'id': int(x['filename'][-16:-4]), 'label': x['label']} for x in meta]
    b = [(x['id'], x['label']) for x in a]

    np.savez_compressed('WhaleSharkMeta.npz', classes=b, meta=a)

    # caculate GT matrix (Used for evaluation ONLY)
    ti = time.time()
    print('[creating GT matrix...]', end=" ", flush=True)
    gt_s_ij = np.zeros((features.shape[0], features.shape[0]))
    cats = []
    for i, sample1 in enumerate(meta):
        if sample1['label'] not in cats:
            cats.append(sample1['label'])
        for j, sample2 in enumerate(meta):
            if sample1['label'] == sample2['label']:
                gt_s_ij[i,j] = 1
    n_cats = len(cats)
    print('done %s [%.1fs]'%(str(gt_s_ij.shape), time.time() - ti))

    np.savez_compressed('WhaleSharkGT.npz', gt_similarity = gt_s_ij)

    # calculate similarity matrix (from features)
    # create a list of lists with feats. per class (C, (N, D))
    ti = time.time()
    print('[calculating similarity matrix (%s)...]'%args.metric, end=" ", flush=True)
    feats = [ [] for _ in range(n_cats)] # create list with empty lists
    current_label = meta[0]['label']
    cont = 0
    for i, feat in enumerate(features):
        if current_label != meta[i]['label']:
            cont += 1
            current_label = meta[i]['label']
        feats[cont].append(feat)
    s_ij = calculate_similarity(feats, metric=args.metric)
    print('done %s [%.1fs]'%(str(s_ij.shape), time.time() - ti))

    np.savez_compressed('WhaleSharkGT.npz', gt_similarity = s_ij)