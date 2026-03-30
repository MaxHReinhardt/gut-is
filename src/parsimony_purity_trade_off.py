import pandas as pd
import numpy as np
from math import log2
from itertools import product
from src.clustering import leiden_clustering, spectral_clustering, agglomerative_clustering, apply_threshold_to_similarity_matrix, calculate_ami, calculate_f1


def evaluate_relation_entropy(df_clusters, df_relations):
    """
    Calculates the weighted average Shannon entropy of relations between constructs of distinct clusters.

    Args:
        df_clusters: df that assigns constructs to clusters, expected to have columns "construct_id" and "cluster_id"
        df_relations: df that contains the relations between constructs, expected to have columns "construct_id_1", "construct_id_2", "path_coefficient".
            Note that the function does not filter out insignificant relations, this must be done before, if desired.
    
    Returns:
        weighted average entropy score
    """

    # Ensure that all path coefficients are valid
    if not df_relations["path_coefficient"].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x)).all():
        raise ValueError("Invalid path coefficients found.")

    # Merge cluster IDs into relation df
    df = df_relations.merge(
        df_clusters.rename(columns={"construct_id": "construct_id_1", "cluster_id": "cluster_1"}),
        on="construct_id_1"
    ).merge(
        df_clusters.rename(columns={"construct_id": "construct_id_2", "cluster_id": "cluster_2"}),
        on="construct_id_2"
    )

    # Keep only inter-cluster relations
    df = df[df["cluster_1"] != df["cluster_2"]]

    # Return 1.0 if all constructs are in the same cluster
    if df.empty:
        return 1.0

    # Compute entropy per cluster_pair
    entropy_by_pair = {}
    counts_by_pair = {}

    for pair, group in df.groupby(["cluster_1", "cluster_2"]):
        num_pos_relations = len(group[group["path_coefficient"] >= 0])
        num_neg_relations = len(group[group["path_coefficient"] < 0])
        pos_prob = num_pos_relations / (num_pos_relations + num_neg_relations)
        neg_prob = num_neg_relations / (num_pos_relations + num_neg_relations)
        
        # Entropy calculation, avoiding errors due to taking the log of 0
        entropy = 0
        if pos_prob > 0:
            entropy -= pos_prob * log2(pos_prob)
        if neg_prob > 0:
            entropy -= neg_prob * log2(neg_prob)

        entropy_by_pair[pair] = entropy
        counts_by_pair[pair] = num_pos_relations + num_neg_relations

    # Weighted average entropy
    total_relations = sum(counts_by_pair.values())
    weighted_entropy = sum(entropy_by_pair[p] * counts_by_pair[p] for p in entropy_by_pair) / total_relations

    return weighted_entropy


def calculate_intra_cluster_dissimilarity(df_clusters, similarity_matrix_df):
    """
    Calculates the average dissimilarity of a construct to all other constructs in the same cluster.

    Args:
        df_clusters: df that assigns constructs to clusters, expected to have columns "construct_id" and "cluster_id"
        similarity_matrix_df: Construct similarity matrix, values are expected to be in [0, 1]
    
    Returns:
        average dissimilarity score
    """

    weighted_similarity = 0
    total_weights = 0

    # Iterate over clusters
    for _, members in df_clusters.groupby('cluster_id')['construct_id']:
        cluster_size = len(members)
        if cluster_size > 1:
            # Subset the similarity matrix for members of this cluster
            submat = similarity_matrix_df.loc[members, members]
            
            # Compute average intra-cluster similarity as the mean of the lower-triangular similarity matrix of the cluster
            sims = np.tril(submat.values, k=-1)
            num_pairs = (cluster_size * (cluster_size - 1)) / 2
            avg_sim =  sims.sum() / num_pairs

            weighted_similarity += (avg_sim * cluster_size)
            total_weights += cluster_size

    if total_weights > 0:
        avg_weighted_similarity = weighted_similarity / total_weights
    else: 
        avg_weighted_similarity = 1.0  # Set similarity to 1.0 when each construct forms its own cluster, resulting in 0.0 dissimilarity

    # 1 - similarity yields dissimilarity
    avg_weighted_dissimilarity = 1.0 - avg_weighted_similarity

    return avg_weighted_dissimilarity


def calculate_cluster_to_construct_ratio(df_clusters):
    """
    Calculates the parsimony loss.

    Args:
        df_clusters: df that assigns constructs to clusters, expected to have columns "construct_id" and "cluster_id"

    Returns: 
        num_clusters / num_constructs
    """

    num_clusters = df_clusters["cluster_id"].nunique()
    num_constructs = df_clusters["construct_id"].nunique()
    
    if num_constructs == 0:
        return 0  # Avoid division by zero
    
    return num_clusters / num_constructs


def cluster_grid_search(cluster_method, similarity_threshold_list, cluster_parameter_list, similarity_df, df_relations=None):
    """
    Calculates the parsimony and purity losses across a grid of cluster solutions for a given algorithm.

    Args: 
        cluster_method: 'leiden', 'spectral', or 'agglomerative'
        similarity_threshold_list: List of threshold values considered for denoising the similarity matrix
        cluster_parameter_list: List of values considered for the parameter of the clustering algorithm that determines the number of clusters
            (resolution for Leiden, the explicit number of clusters for spectral and agglomerative clustering).
        similarity_df: Similarity matrix
        df_relations: df that contains the relations between constructs, expected to have columns "construct_id_1", "construct_id_2", "path_coefficient".
            If None, the relation purity loss calculation is omitted.

    Returns:
        df containing the loss values for each configuration in the search grid
    """

    results = []
    for threshold, cluster_parameter in product(similarity_threshold_list, cluster_parameter_list):
        # Apply thresholding
        similarity_df_th = apply_threshold_to_similarity_matrix(similarity_df=similarity_df, similarity_threshold=threshold)
        
        # Perform clustering
        if cluster_method == 'leiden':
            cluster_df = leiden_clustering(affinity_matrix_df=similarity_df_th, resolution=cluster_parameter)
        elif cluster_method == 'spectral':
            cluster_df = spectral_clustering(affinity_matrix_df=similarity_df_th, n_clusters=cluster_parameter)
        elif cluster_method == 'agglomerative':
            cluster_df = agglomerative_clustering(affinity_matrix_df=similarity_df_th, n_clusters=cluster_parameter)
        else:
            raise ValueError('cluster method not valid.')
        
        # Calculate scores for cluster solution
        if df_relations is not None:
            weighted_entropy = evaluate_relation_entropy(df_clusters=cluster_df, df_relations=df_relations)
        else:
            weighted_entropy = 999  # placeholder
        relative_num_clusters = calculate_cluster_to_construct_ratio(df_clusters=cluster_df)
        avg_weighted_dissimilarity = calculate_intra_cluster_dissimilarity(df_clusters=cluster_df, similarity_matrix_df=similarity_df_th)

        results.append((cluster_method, threshold, cluster_parameter, weighted_entropy, relative_num_clusters, avg_weighted_dissimilarity))

    results_df = pd.DataFrame(
        results,
        columns=["cluster_method", "threshold", "cluster_parameter", "weighted_entropy", "relative_num_clusters", "weighted_dissimilarity"]
    )
    
    return results_df


def calculate_total_loss(purity_score, relative_num_clusters, alpha):
    """
    Calculates the total loss as a convex combination of the parsimony and purity losses.

    Args: 
        purity_score: Purity loss value
        relative_num_clusters: Parsimony loss score
        alpha: Weight parameter, higer values prioritize purity over parsimony and vice versa. 
    
    Returns: 
        Total loss value
    """

    for var_name, var_value in {"purity_score": purity_score, "relative_num_clusters": relative_num_clusters, "alpha": alpha}.items():
        if not (0 <= var_value <= 1):
            raise ValueError(f"{var_name} must be between 0 and 1. Got {var_value}.")

    return alpha * purity_score + (1 - alpha) * relative_num_clusters


def return_best_cluster_solution(alpha, purity_loss_type, results_df, similarity_df, seed):
    """
    Returns the best cluster solution for a chosen balance of parsimony and purity. First, the best algorithmic configuration for the chosen alpha is identified, based 
    on a previously conducted grid search. Following, a cluster solution is re-computed using the best configuration.

    Args:
        alpha: Weight parameter balancing parsimony and purity. Higer values prioritize purity over parsimony and vice versa. 
        purity_loss_type: Determines whether the total loss is calculated using the relation or construct purity loss ('relation_entropy' or 'intra_cluster_dissimilarity')
        results_df: Output from function cluster_grid_search
        similarity_df: Similarity matrix
        seed: Random seed
    
    Returns: 
        Dict with information on the best configuration
        df with cluster solution
    """

    if not (0 <= alpha <= 1):
        raise ValueError(f"Alpha must be between 0 and 1. Got {alpha}.")
    
    if purity_loss_type not in ['relation_entropy', 'intra_cluster_dissimilarity']:
        raise ValueError(f"Invalid purity_loss_type.")

    # Calculate the total loss for each row
    results_df = results_df.copy()
    results_df["total_loss"] = results_df.apply(
        lambda row: calculate_total_loss(
            purity_score=row["weighted_entropy"] if purity_loss_type == 'relation_entropy' else row["weighted_dissimilarity"],
            relative_num_clusters=row["relative_num_clusters"],
            alpha=alpha),
        axis=1)

    # Find the row with minimum total loss
    best_row = results_df.loc[results_df["total_loss"].idxmin()]

    best_cluster_method = best_row["cluster_method"]
    best_threshold = best_row["threshold"]
    best_cluster_parameter = best_row["cluster_parameter"]
    best_weighted_dissimilarity = best_row["weighted_dissimilarity"]
    best_entropy = best_row["weighted_entropy"]
    best_relative_num_clusters = best_row["relative_num_clusters"]
    best_total_loss = best_row["total_loss"]

    # Recompute the cluster solution with the best parameters
    similarity_df_th = apply_threshold_to_similarity_matrix(similarity_df=similarity_df, similarity_threshold=best_threshold)

    if best_cluster_method == 'leiden':
        best_cluster_df = leiden_clustering(
            affinity_matrix_df=similarity_df_th, 
            resolution=best_cluster_parameter,
            seed=seed
            )
    elif best_cluster_method == 'spectral':
        best_cluster_df = spectral_clustering(
            affinity_matrix_df=similarity_df_th,
            n_clusters=best_cluster_parameter,
            seed=seed
            )
    elif best_cluster_method == 'agglomerative':
        best_cluster_df = agglomerative_clustering(
            affinity_matrix_df=similarity_df_th,
            n_clusters=best_cluster_parameter
            )
    else:
        raise ValueError('Best cluster method not valid.')

    return {
        "best_cluster_method": best_cluster_method,
        "best_threshold": best_threshold,
        "best_resolution": best_cluster_parameter,
        "best_weighted_dissimilarity": best_weighted_dissimilarity,
        "best_entropy": best_entropy,
        "best_relative_num_clusters": best_relative_num_clusters,
        "total_loss": best_total_loss,
    }, best_cluster_df


def calculate_cluster_stats_across_alpha(grid_search_df, similarity_df, true_cluster_df=None, purity_loss_type='intra_cluster_dissimilarity', 
                                         num_alpha_values=101, num_repetitions=1):
    """
    Calculates cluster stats across a number of alpha values in [0, 1], including AMI and F1 compared against a ground truth, as well as the parsimony and purity losses.

    Args:
        grid_search_df: Output from function cluster_grid_search
        similarity_df: Similarity matrix
        true_cluster_df: df containing the ground truth clusters with construct IDs in the first and cluster IDs in the second column. If None, AMI and F1 are omitted.
        purity_loss_type: Determines whether the total loss is calculated using the relation or construct purity loss ('relation_entropy' or 'intra_cluster_dissimilarity')
        num_alpha_values: Number of considered alpha values in [0, 1]
        num_repetitions: Number of runs for each alpha (more repetitions increase the quality of estimates for non-deterministic cluster algorithms)

    Returns: 
        Result dict
    """

    # Initialize arrays to store results
    ami_values = np.zeros((num_repetitions, num_alpha_values))
    f1_values = np.zeros((num_repetitions, num_alpha_values))
    f1_baseline_values = np.zeros((num_repetitions, num_alpha_values))
    weighted_dissimilarities = np.zeros((num_repetitions, num_alpha_values))
    weighted_entropies = np.zeros((num_repetitions, num_alpha_values))
    relative_num_cluster_values = np.zeros((num_repetitions, num_alpha_values))

    alpha_values = np.linspace(0, 1.0, num_alpha_values)

    for rep in range(num_repetitions):
        for i, alpha in enumerate(alpha_values):
            stats, cluster_solution = return_best_cluster_solution(
                alpha=alpha, 
                purity_loss_type=purity_loss_type, 
                results_df=grid_search_df, 
                similarity_df=similarity_df,
                seed=rep
            )
            if true_cluster_df is not None:
                ami = calculate_ami(true_cluster_df=true_cluster_df, pred_cluster_df=cluster_solution)
                f1_results = calculate_f1(true_cluster_df=true_cluster_df, pred_cluster_df=cluster_solution, baseline_similarity_df=similarity_df)
            else: 
                ami = None
                f1_results = {'f1_pred': None, 'best_f1_baseline': None, 'best_threshold_baseline': None}

            ami_values[rep, i] = ami
            f1_values[rep, i] = f1_results["f1_pred"]
            f1_baseline_values[rep, i] = f1_results["best_f1_baseline"]
            weighted_dissimilarities[rep, i] = stats["best_weighted_dissimilarity"]
            weighted_entropies[rep, i] = stats["best_entropy"]
            relative_num_cluster_values[rep, i] = stats["best_relative_num_clusters"]

    # Extract only the first F1 baseline value, since its calculation is deterministic
    first_baseline_value = f1_baseline_values[0, 0]

    return {
        "ami_values": ami_values, 
        "f1_values": f1_values,
        "baseline_f1": first_baseline_value, 
        "weighted_dissimilarities": weighted_dissimilarities, 
        "weighted_entropies": weighted_entropies, 
        "relative_num_cluster_values": relative_num_cluster_values
    }

