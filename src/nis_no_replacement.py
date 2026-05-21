import numpy as np
from random import choices

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

def no_resampling_nis(gt_similarity, similarity, N_v, N_n, n_hat=None, ci=False, threshold=0.95):

    NUM_IMAGES = len(gt_similarity)
    thresholded_sim = similarity > threshold

    # # estimate the degree using similarity scores
    # if not n_hat: 
    #     n_hat = []
    #     for i in range(NUM_IMAGES):
    #         n_hat.append(np.sum(similarity[i]))
    #     n_hat = np.power(n_hat, -1)

    # # For some reason this needs to run everytime.
    # n_hat = np.array(n_hat)

    # # sample vertices without replacement and inversely proportional to the estimate degree above
    # Q = [(n_hat[i], i) for i in range(len(n_hat))]
    # sampled_vertices = MaxGumbel(Q, N_v)
    # total_pool_weight = np.sum(n_hat)

    # probability_at_sampling_vertex = n_hat[sampled_vertices] * len(sampled_vertices) / total_pool_weight

    # # this is needed for the exception later
    # sampled_vertices = list(sampled_vertices)

    # Sample uniformly instead
    sampled_vertices = list(np.random.choice(range(len(n_hat)), N_v, replace=False))
    probability_at_sampling_vertex = 1 / np.array(range(NUM_IMAGES, NUM_IMAGES - len(sampled_vertices), -1))

    # initialize the cache
    cache = []
    for image in sampled_vertices:
        cache.append([[image, image, 1]])
    
    estimated_num_individuals = 0

    # for each sampled vertex we build an estimate of the degree
    for j, image in enumerate(sampled_vertices):
        images_to_remove = [image]
        number_known_neighbors = 1

        # check the cache to see if there are any already known neighbors for image
        for edge in cache[j][1:]:
            if edge[0] == image:
                images_to_remove.append(edge[1])
            else:
                images_to_remove.append(edge[0])

            if edge[2] == 1:
                number_known_neighbors += 1 

        # sample neighbors proportional to the average similarity to the thresholded neighbors of vertex with multiplicative gating by the similarity to image
        high_confidence_neighbors = np.where(thresholded_sim[image] == 1)[0]

        # Multiplicative Gating
        if len(high_confidence_neighbors) > 1:
            q = [(similarity[image][x] * (sum(similarity[x][high_confidence_neighbors]) - similarity[image][x]) / (len(high_confidence_neighbors) - 1), x) for x in range(len(similarity[image]))]
            q_trimmed =  [x for x in q if x[0] not in images_to_remove]

            sampled_neighbors = MaxGumbel([x for x in q_trimmed], N_n)

            probability_at_sampling_neighbor = np.array([float(x[0]) for x in q])[sampled_neighbors]
            total_pool_weight = sum(probability_at_sampling_neighbor)
            probability_at_sampling_neighbor *= len(sampled_neighbors)
            probability_at_sampling_neighbor /= total_pool_weight

            estimated_degree_final = number_known_neighbors

        else:
            # sample neighbors of image proportional to the similarity
            q = [(similarity[image][i], i) for i in range(len(similarity[image])) if i not in images_to_remove]
            total_pool_weight = sum([similarity[image][i] for i in range(len(similarity[image])) if i not in images_to_remove])
            sampled_neighbors = MaxGumbel(q, N_n)
            probability_at_sampling_neighbor = similarity[image][sampled_neighbors] * len(sampled_neighbors) / total_pool_weight

        estimated_degree_final = number_known_neighbors

        # for each neighbor sampled
        for i, neighbor in enumerate(sampled_neighbors):

            estimated_degree = 0
            
            # if this edge is relevant to any of the other sampled vertices then add it to the cache (remember there are no duplicate vertices)
            try:
                cache[sampled_vertices.index(neighbor)].append([neighbor, image, gt_similarity[neighbor][image]])
            except:
                pass

            # Marginal inclusion probability approximation
            pi_k = probability_at_sampling_neighbor[i]
            pi_k = min(pi_k, 1.0)
            estimated_degree += gt_similarity[image][neighbor] / pi_k

            estimated_degree_final += estimated_degree

        # running sum of estimated number of individuals to be scaled by the number of sampled vertices in the end

        pj = probability_at_sampling_vertex[j]
        pj = min(pj, 1.0)
        estimated_num_individuals += 1/(estimated_degree_final*pj)

    estimated_num_individuals /= len(sampled_vertices)

    return estimated_num_individuals, n_hat