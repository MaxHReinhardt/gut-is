from src.parsimony_purity_trade_off import cluster_grid_search, calculate_cluster_stats_across_alpha
from src.clustering import get_similarity_matrix_from_embeddings
from src.projection_model import ProjectionHead
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# Paths
true_cluster_csv = ''  # CSV containing the ground truth clusters with construct IDs in the first and cluster IDs in the second column
relation_csv = None  # CSV that contains the relations between constructs, expected to have columns "construct_id_1", "construct_id_2", "path_coefficient". Only needed when 'relation_entropy' is used as purity loss.
model_path = ''  # Path of the state dict of the projection model
emb_folders=['']  # List of directory paths, containing the embeddings for the constructs in construct_df

# Model settings
input_dim = 4096
proj_dim = 2048
device = 'cpu'

# Clustering settings
cluster_method = 'agglomerative'
threshold_list = [x for x in np.linspace(0, 0.95, 51)]  # Thresholds for denoising the similarity matrix
cluster_param_list = [int(round(x)) for x in np.linspace(1, 402, 51)]  # Number of clusters for agglomerative and spectral clustering, resolution for Leiden

# Continuum exploration settings
purity_loss_type = 'intra_cluster_dissimilarity'  # alternatively 'relation_entropy'
num_alpha_values = 21  # Number of alpha values in [0, 1] to explore
num_repetitions = 1  # Number of runs for each alpha. For agglomerative clustering, 1 is sufficient as the method is deterministic.


if __name__ == '__main__':
    # Load data and model
    true_cluster_df = pd.read_csv(true_cluster_csv)
    if relation_csv is not None:
        relation_df = pd.read_csv(relation_csv)

    model = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get a list of construct IDs
    construct_ids = true_cluster_df.iloc[:, 0].tolist()

    # Calculate similarity matrix with projection model
    similarity_df = get_similarity_matrix_from_embeddings(
        id_list=construct_ids, 
        emb_folders=emb_folders, 
        model=model, 
        device=device
    )

    # Perform grid search
    results_df_agglomerative = cluster_grid_search(
        cluster_method=cluster_method, 
        similarity_threshold_list=threshold_list, 
        cluster_parameter_list=cluster_param_list, 
        similarity_df=similarity_df, 
        df_relations=relation_df if relation_csv is not None else None)
    
    cluster_stats = calculate_cluster_stats_across_alpha(
        grid_search_df=results_df_agglomerative, 
        similarity_df=similarity_df, 
        true_cluster_df=true_cluster_df, 
        purity_loss_type=purity_loss_type,
        num_alpha_values=num_alpha_values,
        num_repetitions=num_repetitions)
    
    # Compute alpha values
    alpha_values = np.linspace(0, 1.0, 21)

    # Extract and print maximum F1 and AMI scores
    mean_amis = cluster_stats["ami_values"].mean(axis=0)
    max_idx = np.argmax(mean_amis)  # Index of maximum AMI
    print("Max AMI:", mean_amis[max_idx], 'occured at alpha = ', alpha_values[max_idx])

    mean_f1s = cluster_stats["f1_values"].mean(axis=0)
    max_idx = np.argmax(mean_f1s)  # Index of maximum F1
    print("Max F1:", mean_f1s[max_idx], 'occured at alpha = ', alpha_values[max_idx])

    # Compute means of losses for each alpha
    intra_purity_mean = cluster_stats["weighted_dissimilarities"].mean(axis=0)
    parsimony_mean = cluster_stats["relative_num_cluster_values"].mean(axis=0)
    if relation_csv is not None: 
        relation_purity_mean = cluster_stats["weighted_entropies"].mean(axis=0)

    # Create plot of losses across alphas
    plt.figure(figsize=(7, 4))
    if relation_csv is not None:
        l1 = plt.plot(alpha_values, relation_purity_mean, label=r"$L_{\text{relation\_purity}}$", marker='o')[0]
    l2 = plt.plot(alpha_values, intra_purity_mean, label=r"$L_{\text{construct\_purity}}$", marker='s')[0]
    l3 = plt.plot(alpha_values, parsimony_mean, label=r"$L_{\text{parsimony}}$", marker='^')[0]
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper left')  
    plt.show()
