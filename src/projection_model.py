import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class ProjectionHead(nn.Module):
    """
    Linear projection head to map text embeddings into a task-specific space.
    """
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x):
        x = self.linear(x)
        x = F.normalize(x, dim=-1)
        return x


class PairwiseEmbeddingDataset(Dataset):
    """
    Dataset class, providing pairs of embeddings and their ground truth similarity.
    
    Args:
        pairs_table: CSV or df with columns "variableId1", "variableId2", "similarity". Each row reflects the ground truth similarity for a pair of constructs;
            similarity scores are expected to be binary.
        emb_folders: List of folders that contain embedding files (format: "{id}.pt") for each ID in pairs_table. If multiple folders are provided, the embeddings
            for each construct are concatenated to form a joint input.
    """
    def __init__(self, pairs_table, emb_folders):
        # Load the pairs table
        if isinstance(pairs_table, str):
            df = pd.read_csv(pairs_table)
        else:
            df = pairs_table.copy()

        # Ensure that negative class is indicated by -1
        df['similarity'] = df['similarity'].replace(0, -1)

        self.construct_id_1 = df["variableId1"].astype(str).tolist()
        self.construct_id_2 = df["variableId2"].astype(str).tolist()
        self.similarity = torch.tensor(df["similarity"].values, dtype=torch.float32)

        # Preload embeddings
        unique_ids = set(df['variableId1']).union(df['variableId2'])
        self.embeddings = {}
        for id in unique_ids:
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
            self.embeddings[str(id)] = embs_cat

    def __len__(self):
        return len(self.similarity)

    def __getitem__(self, idx):
        id1 = self.construct_id_1[idx]
        id2 = self.construct_id_2[idx]
        sim = self.similarity[idx]
        emb1 = self.embeddings[str(id1)]
        emb2 = self.embeddings[str(id2)]
        return emb1, emb2, sim
    

class BalancedPairwiseEmbeddingDataset(Dataset):
    """
    Dataset class, providing pairs of embeddings and their ground truth similarity. This version balances positives and negatives by using hard negative sampling.
    
    Args:
        pairs_table: CSV or df with columns "variableId1", "variableId2", "similarity". Each row reflects the ground truth similarity for a pair of constructs;
            similarity scores are expected to be binary.
        emb_folders: List of folders that contain embedding files (format: "{id}.pt") for each ID in pairs_table. If multiple folders are provided, the embeddings
            for each construct are concatenated to form a joint input.
        neg_ratio: Number of negatives per positive
        subset_fraction: Fraction of negatives to consider for hard sampling per epoch (lower values may be chosen for efficiency)
    """
    def __init__(self, pairs_table, emb_folders, neg_ratio=1.0, subset_fraction=1.0):
        if isinstance(pairs_table, str):
            df = pd.read_csv(pairs_table)
        else:
            df = pairs_table.copy()

        # Ensure that negative class is indicated by -1
        df['similarity'] = df['similarity'].replace(0, -1)

        # Separate positives and negatives
        self.pos_df = df[df['similarity'] == 1].reset_index(drop=True)
        self.neg_df = df[df['similarity'] == -1].reset_index(drop=True)
        self.neg_ratio = neg_ratio
        self.subset_fraction = subset_fraction

        # Preload embeddings
        unique_ids = set(df['variableId1']).union(df['variableId2'])
        self.embeddings = {}
        for id in unique_ids:
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
            self.embeddings[str(id)] = embs_cat

        # Placeholder for current samples, containing all positive instances and current negatives (updated each epoch)
        self.current_samples = []

    def sample_negatives(self, model, device='cpu', margin=0.0, delta=0.05):
        """
        Sample a mix of hard (50%), semi-hard (30%), and easy (20%) negatives. Only uses a random subset of negatives for efficiency.

        Args: 
            model: Current embedding model used to identify hard, semi-hard, and easy negatives
            device: PyTorch device
            margin: Margin parameter (m), negatives above the margin are considered hard
            delta: Separates semi-hard and easy negatives. Negatives > m-delta are semi-hard, negatives <= m-delta are easy.
        """
        # Random subset of negatives
        n_subset = max(1, int(len(self.neg_df) * self.subset_fraction))
        neg_subset = self.neg_df.sample(n_subset, random_state=random.randint(0, 1_000_000)).reset_index(drop=True)

        model.eval()
        sims = []

        # Compute similarities in batches to avoid large tensor allocation
        for start in range(0, len(neg_subset), 256):
            end = start + 256
            batch_ids1 = neg_subset['variableId1'][start:end]
            batch_ids2 = neg_subset['variableId2'][start:end]

            # Prepare batch embeddings
            batch_x1 = torch.cat([self.embeddings[idx] for idx in batch_ids1], dim=0).to(device)
            batch_x2 = torch.cat([self.embeddings[idx] for idx in batch_ids2], dim=0).to(device)

            with torch.no_grad():
                z1 = model(batch_x1)
                z2 = model(batch_x2)
                batch_sim = F.cosine_similarity(z1, z2)
                sims.append(batch_sim.cpu())

        # Concatenate all batch similarities
        sims = torch.cat(sims).numpy()
        neg_subset['sim'] = sims

        # Sort by similarity
        neg_subset = neg_subset.sort_values('sim', ascending=False).reset_index(drop=True)

        # Define bins relative to margin
        hard = neg_subset[neg_subset['sim'] > margin]
        semi_hard = neg_subset[(neg_subset['sim'] <= margin) & (neg_subset['sim'] > margin - delta)]
        easy = neg_subset[neg_subset['sim'] <= margin - delta]

        # Sample negatives in fixed ratios
        n_total = int(len(self.pos_df) * self.neg_ratio)
        n_hard = min(len(hard), int(0.5 * n_total))
        n_semi_hard = min(len(semi_hard), int(0.3 * n_total))
        n_easy = min(len(easy), n_total - n_hard - n_semi_hard)

        print(f"positive: {len(self.pos_df)}, hard: {n_hard}, semi: {n_semi_hard}, easy: {n_easy}.")

        current_negatives = pd.concat([
            hard.sample(n_hard) if n_hard > 0 else pd.DataFrame(),
            semi_hard.sample(n_semi_hard) if n_semi_hard > 0 else pd.DataFrame(),
            easy.sample(n_easy) if n_easy > 0 else pd.DataFrame()
        ]).reset_index(drop=True)

        # Combine positives and sampled negatives, then shuffle
        self.current_samples = pd.concat([self.pos_df, current_negatives]).sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.current_samples)

    def __getitem__(self, idx):
        row = self.current_samples.iloc[idx]
        id1 = row['variableId1']
        id2 = row['variableId2']
        sim = row['similarity']
        emb1 = self.embeddings[id1]
        emb2 = self.embeddings[id2]
        return emb1, emb2, torch.tensor(sim, dtype=torch.float32)
    

def evaluate_auc(model, dataset, device="cpu", batch_size=256):
    """
    Evaluates the ROC-AUC of an embedding model.

    Args: 
        model: Projection model. If None, embeddings from dataset are used without projection.
        dataset: Evaluation data, expected to be PairwiseEmbeddingDataset.
        device: PyTorch device
        batch_size: Batch size

    Return: 
        ROC-AUC score
    """
    if model is not None: 
        model.eval()
        model.to(device)

    sims = []
    labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # If a model is provided, apply projection
            if model is not None:
                z1 = model(x1)
                z2 = model(x2)
            # Else flatten all dimensions, except for batch dim, to calculate similarity of original embeddings
            else:
                z1 = x1.flatten(start_dim=1)
                z2 = x2.flatten(start_dim=1)

            # Compute cosine similarity for each pair
            cos_sim = F.cosine_similarity(z1, z2)
            sims.extend(cos_sim.cpu().numpy())
            labels.extend(y.cpu().numpy())

    # Ensure labels are in {0, 1}
    labels = np.array(labels)
    labels = np.where(labels == -1, 0, labels)

    auc = roc_auc_score(labels, sims)
    
    print(f"AUC: {auc:.4f}")

    return auc
    

def train_projection_head(model, train_set, val_set=None, epochs_step_size=1, batch_size=64, epochs=30, lr=1e-4, weight_decay=0, 
                          margin=0.1, delta=0.05, device="cpu"):
    """
    Training function for the projection head. The model is optimized using CosineEmbeddingLoss and Adam optimizer.

    Args:
        model: Initialized projection head model to be trained
        train_set: Train data, expected to be BalancedPairwiseEmbeddingDataset
        val_set: Validation data, expected to be PairwiseEmbeddingDataset. If None, validation performance is not evaluated during training.
        epochs_step_size: Number of epochs after which the model is evaluated on the validation data
        batch_size: Size of training batches
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2-regularization weight
        margin: Margin parameter for contrastive loss and hard negative sampling
        delta: Delta parameter for hard negative sampling
        device: PyTorch device
    
    Returns:
        Trained model
        List of validation AUCs
    """
    val_aucs = []

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CosineEmbeddingLoss(margin=margin)

    for epoch in range(epochs):
        # Sample negatives at the start of each epoch
        train_set.sample_negatives(model=model, device=device, margin=margin, delta=delta)
        dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

        model.train()
        total_loss = 0.0

        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x1.size(0)  # CosineEmbeddingLoss returns mean over batch

        avg_loss = total_loss / len(train_set)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if val_set is not None and (epoch + 1) % epochs_step_size == 0:
            auc = evaluate_auc(model=model, dataset=val_set, device=device, batch_size=256)
            val_aucs.append(auc)

    return model, val_aucs
