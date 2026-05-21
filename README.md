# Hector's Synthesis Project

Some dependencies that I did not have installed (you all will probably have this): torch, torchvision

## Prepare the Data

In order to save some time when experimenting with the data, follow the steps below to download the whale shark data and classes, compute features, compute cosine similarity, and generate .npz files (totalling < 1gb) to save all the relevant information.

1) python compute_features.py --dataset WhaleSharkID
2) python process_data.py --dataset WhaleSharkID

Note: On CPU this will take around an hour (according to the original GitHub), on GPU the load and compute is much faster, around 10 minutes (on my 5070ti). 

## Visualize the Data

If you would like to familiarize yourself with the database, then visualize.py is your friend. Running the following code will generate a pair of drawings representing an isolated part of the total data set. The variable 'n' in the main function of visualize.py is the number of randomly chosen individuals you would like to visualize (should be fairly small since each individual will already have many nodes and edges). The first drawing will show the ground truth graph for the 'n' isolated individuals, and the second drawing will show the similarity graph on those same individuals.

1) python visualize.py

visualize.py also has functionality to show the value of the clique collapsing idea. There is a commented block in main, which when uncommented will generate a third drawing. The third drawing shows how clique collapse changes the similarity graph for the selected individuals.

## Compare NIS and NIS Without Replacement

Running the following will generate a plot comparing NIS (taken from the original GitHub with no changes) to NIS without replacement. The NIS without replacement methods can be found in src/nis_no_replacement.py.

You can modify the parameters of the run by changing the arguments to NIS() in the main function of compare.py. 'runs' is the number of runs averaged over, max_N is the largest value N, from which we derive the number of vertices and neighbors of vertices to sample. Good values are [102, 201, 300]. Because of how the code is written, there is not much scaling slowdown in increasing max_N. The cache is designed to be as small as possible and efficient to draw and update from as I could think of. I point this out because the clique cache grew very fast and would see significant slowdown at bigger values of max_N. ratio is the ratio of vertices sampled to neighbors of vertices sampled.

1) python compare.py