import os
from itertools import combinations
import pandas as pd
import numpy as np
import igraph as ig
import leidenalg
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_mutual_info_score, f1_score


def get_similarity_matrix_from_embeddings(id_list, emb_folders, model=None, device="cpu"):
    """
    Calculates the cosine similarity between provided embeddings to produce a similarity matrix. Optionally, the embeddings are projected in a 
    (task-specific) space before similarities are calculated.

    Args:
        id_list: List of construct IDs for which the similarity matrix should be created
        emb_folders: List of folders that contain embedding files (format: "{id}.pt") for each ID in pairs_table. If multiple folders are provided, the embeddings
            for each construct are concatenated to form a joint input.
        model: Projection model. If None, embeddings from dataset are used without projection.
        device: PyTorch device

    Returns: 
        similarity df
    """

    if model is not None: 
        model.eval()
        model.to(device)
    
    # Similar to processing in dataset classes
    embeddings = []
    for id in id_list:
        embs = []
        for emb_folder in emb_folders:
            path = os.path.join(emb_folder, f"{id}.pt")
            emb = torch.load(path, map_location="cpu").float()
            if emb.ndim > 1:
                emb = emb.squeeze()
            emb = F.normalize(emb, dim=-1)
            embs.append(emb)
            
        embs_cat = torch.cat(embs, dim=-1)  # concatenate along features
        embs_cat = embs_cat.unsqueeze(0)

        if model is not None:
            with torch.no_grad():
                embedding = model(embs_cat.to(device))
        else:
            embedding = embs_cat
        embeddings.append(embedding)
    
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    similarity_matrix = cosine_similarity(embeddings)

    # Normalize cosine similarities to interval [0, 1], as required for some clustering methods (e.g. Leiden)
    min_val, max_val = similarity_matrix.min(), similarity_matrix.max()
    similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)

    # To pandas df
    similarity_df = pd.DataFrame(similarity_matrix, index=id_list, columns=id_list)

    return similarity_df


def apply_threshold_to_similarity_matrix(similarity_df, similarity_threshold=0.5):
    return similarity_df.where(similarity_df >= similarity_threshold, 0.0)


def leiden_clustering(affinity_matrix_df, resolution, seed=42):
    """
    Performs Leiden clustering using the leidenalg implementation.

    Args: 
        affinity_matrix_df: Similarity matrix
        resolution: Resolution parameter for Leiden clustering. 
        seed: Random seed

    Returns:
        df with cluster assignments for all construct IDs
    """

    # Build an igraph graph from the similarity matrix
    edges = []
    weights = []

    for i, c1 in enumerate(affinity_matrix_df.index):
        for j, c2 in enumerate(affinity_matrix_df.columns):
            if j <= i:
                continue  # avoid duplicates and self-loops
            weight = affinity_matrix_df.at[c1, c2]
            if weight > 0:  # consider only positive weights
                edges.append((i, j))
                weights.append(weight)

    num_nodes = len(affinity_matrix_df)
    g = ig.Graph(n=num_nodes, directed=False)
    g.vs['name'] = list(affinity_matrix_df.index)
    g.add_edges(edges)
    g.es['weight'] = weights

    # Run Leiden clustering
    partition = leidenalg.find_partition(
        graph=g,
        partition_type=leidenalg.RBConfigurationVertexPartition,
        weights=g.es['weight'],
        resolution_parameter=resolution,
        seed=seed
    )

    # Extract cluster membership per node
    labels_leiden = partition.membership

    # Construct the output df
    cluster_df_leiden = pd.DataFrame({
        'construct_id': g.vs['name'],
        'cluster_id': labels_leiden
    })

    return cluster_df_leiden


def spectral_clustering(affinity_matrix_df, n_clusters, seed=42, epsilon=1e-6):
    """
    Performs spectral clustering using the sklearn implementation.

    Args: 
        affinity_matrix_df: Similarity matrix
        n_clusters: Number of clusters
        seed: Random seed
        epsilon: Small value added to all similarities, can stabilize/improve clustering when the affinity matrix is sparse

    Returns:
        df with cluster assignments for all construct IDs
    """

    # Perform spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=seed
    )
    labels = clustering.fit_predict(affinity_matrix_df.values + epsilon)

    # Construct the output df
    cluster_df_spec = pd.DataFrame({
        'construct_id': affinity_matrix_df.index,
        'cluster_id': labels
    })

    return cluster_df_spec


def agglomerative_clustering(affinity_matrix_df, n_clusters):
    """
    Performs agglomerative clustering using the sklearn implementation.

    Args: 
        affinity_matrix_df: Similarity matrix
        n_clusters: Number of clusters

    Returns:
        df with cluster assignments for all construct IDs
    """

    # Convert similarity matrix to distance matrix (1 - similarity)
    distance_matrix = 1 - affinity_matrix_df.values

    # Perform agglomerative clustering
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = agg_clustering.fit_predict(distance_matrix)

    # Construct the output df
    agg_cluster_df = pd.DataFrame({
        'construct_id': affinity_matrix_df.index,
        'cluster_id': labels
    })

    return agg_cluster_df


def calculate_ami(true_cluster_df, pred_cluster_df):
    """
    Calculates the adjusted mutual information (AMI) for a cluster solution and a ground truth.

    Args:
        true_cluster_df: df containing the ground truth clusters with construct IDs in the first and cluster IDs in the second column
        pred_cluster_df: df containing the predicted clusters with construct IDs in the first and cluster IDs in the second column

    Returns:
        AMI
    """

    # Standardize column names
    true_cluster_df = true_cluster_df.rename(columns={
        true_cluster_df.columns[0]: 'construct_id',
        true_cluster_df.columns[1]: 'cluster_id_true'
    })
    pred_cluster_df = pred_cluster_df.rename(columns={
        pred_cluster_df.columns[0]: 'construct_id',
        pred_cluster_df.columns[1]: 'cluster_id_pred'
    })

    # Fill missing cluster IDs with a unique value by hashing the construct ID (constructs without assignment form their own cluster)
    true_cluster_df['cluster_id_true'] = true_cluster_df.apply(
        lambda row: str(hash(row['construct_id'])) if pd.isna(row['cluster_id_true']) or row['cluster_id_true'] == '' else str(row['cluster_id_true']),
        axis=1
    )
    
    # Ensure instance IDs match
    if set(true_cluster_df['construct_id']) != set(pred_cluster_df['construct_id']):
        raise ValueError("Different sets of construct IDs provided.")
    
    # Merge dfs
    merged = pd.merge(true_cluster_df, pred_cluster_df, on='construct_id')
    
    # Compute AMI
    ami = adjusted_mutual_info_score(
        labels_true=merged['cluster_id_true'], 
        labels_pred=merged['cluster_id_pred'], 
        average_method='arithmetic'
    )
    
    return ami


def calculate_f1(true_cluster_df, pred_cluster_df, baseline_similarity_df=None, num_thresholds=51):
    """
    Calculates the F1 score for a cluster solution and a ground truth. Optionally, one can additionally provide a baseline similarity df, 
    e.g. the similarity df before thresholding and clustering. For this similarity matrix, the optimal threshold and corresponding F1 score
    are calculated.

    Args:
        true_cluster_df: df containing the ground truth clusters with construct IDs in the first and cluster IDs in the second column
        pred_cluster_df: df containing the predicted clusters with construct IDs in the first and cluster IDs in the second column
        baseline_similarity_df: Similarity df with continuous similarity scores

    Returns:
        Dict of structure {'f1_pred': score, 'best_f1_baseline': score, 'best_threshold_baseline': value}
    """

    # Standardize column names
    true_cluster_df = true_cluster_df.rename(columns={
        true_cluster_df.columns[0]: 'construct_id',
        true_cluster_df.columns[1]: 'cluster_id'
    })
    pred_cluster_df = pred_cluster_df.rename(columns={
        pred_cluster_df.columns[0]: 'construct_id',
        pred_cluster_df.columns[1]: 'cluster_id'
    })

    # Ensure all construct IDs are strings
    true_cluster_df['construct_id'] = true_cluster_df['construct_id'].astype(str)
    pred_cluster_df['construct_id'] = pred_cluster_df['construct_id'].astype(str)
    if baseline_similarity_df is not None:
        baseline_similarity_df.index = baseline_similarity_df.index.astype(str)
        baseline_similarity_df.columns = baseline_similarity_df.columns.astype(str)
    
    # Ensure construct IDs match
    if set(true_cluster_df['construct_id']) != set(pred_cluster_df['construct_id']):
        raise ValueError("Different sets of construct IDs provided in true and pred.")
    if baseline_similarity_df is not None:
        if set(true_cluster_df['construct_id']) != set(baseline_similarity_df.index.tolist()):
            raise ValueError("Different sets of construct IDs provided in true and baseline.")

    def pairwise_similarities_from_cluster_df(cluster_df):
        """
        Transforms a df with cluster assignments for constructs into a df with pairwise similarities for all possible pairs of constructs.
        """

        # Fill missing cluster IDs with a unique value by hashing the construct ID (constructs without assignment form their own cluster)
        cluster_df['cluster_id'] = cluster_df.apply(
            lambda row: str(hash(row['construct_id'])) if pd.isna(row['cluster_id']) or row['cluster_id'] == '' else str(row['cluster_id']),
            axis=1
        )

        # Map construct_id to cluster_id for fast lookup
        cluster_map = cluster_df.set_index('construct_id')['cluster_id'].to_dict()

        constructs = list(cluster_map.keys())
        pairs = list(combinations(constructs, 2))

        # Convert to numpy arrays for vectorized operations
        construct_id_1 = np.array([p[0] for p in pairs])
        construct_id_2 = np.array([p[1] for p in pairs])

        # Lookup cluster IDs
        cluster_id_1 = np.array([cluster_map[c] for c in construct_id_1])
        cluster_id_2 = np.array([cluster_map[c] for c in construct_id_2])

        # Compute similarities
        similarity = (cluster_id_1 == cluster_id_2).astype(int)

        # Build result df
        similarity_df = pd.DataFrame({
            'construct_id_1': construct_id_1.astype(str),
            'construct_id_2': construct_id_2.astype(str),
            'similarity': similarity
        })

        # Ensure deterministic ordering of constructs to allow for easy merge later
        similarity_df[["construct_id_1", "construct_id_2"]] = pd.DataFrame(
            np.sort(similarity_df[["construct_id_1", "construct_id_2"]].values, axis=1),
            index=similarity_df.index
        )

        return similarity_df
    
    true_pairwise_similarities = pairwise_similarities_from_cluster_df(true_cluster_df)
    pred_pairwise_similarities = pairwise_similarities_from_cluster_df(pred_cluster_df)

    def pairwise_similarities_from_similarity_matrix_df(similarity_matrix_df):
        """
        Extracts unique pairwise similarities from the baseline similarity matrix.
        """
        # Get row and column indices for the lower triangle (excluding diagonal)
        rows, cols = np.tril_indices_from(similarity_matrix_df, k=-1)
        
        # Build df
        similar_pairs_df = pd.DataFrame({
            'construct_id_1': similarity_matrix_df.index[rows].astype(str),
            'construct_id_2': similarity_matrix_df.columns[cols].astype(str),
            'similarity': similarity_matrix_df.values[rows, cols]
        })

        # Ensure deterministic ordering of constructs to allow for easy merge later
        similar_pairs_df[["construct_id_1", "construct_id_2"]] = pd.DataFrame(
            np.sort(similar_pairs_df[["construct_id_1", "construct_id_2"]].values, axis=1),
            index=similar_pairs_df.index
        )
        
        return similar_pairs_df
    
    if baseline_similarity_df is not None:
        baseline_pairwise_similarities = pairwise_similarities_from_similarity_matrix_df(baseline_similarity_df)

    # Merge true, pred, and baseline pairwise similarities
    merged = true_pairwise_similarities.merge(
        pred_pairwise_similarities,
        on=['construct_id_1', 'construct_id_2'],
        suffixes=('_true', '_pred')
    )
    if baseline_similarity_df is not None:
        merged = merged.merge(
            baseline_pairwise_similarities,
            on=['construct_id_1', 'construct_id_2']
        ).rename(columns={'similarity': 'similarity_baseline'})

    # Compute F1 for pred vs true
    f1_pred = f1_score(merged['similarity_true'], merged['similarity_pred'])

    if baseline_similarity_df is not None:
        # Compute F1 for baseline at multiple thresholds
        thresholds = np.linspace(0, 1, num_thresholds)
        best_f1 = 0
        best_threshold = None

        for t in thresholds:
            binary_baseline = (merged['similarity_baseline'] >= t).astype(int)
            f1_t = f1_score(merged['similarity_true'], binary_baseline)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t
    else: 
        best_f1 = None  
        best_threshold = None 

    return {
        'f1_pred': f1_pred,
        'best_f1_baseline': best_f1,
        'best_threshold_baseline': best_threshold
    }



