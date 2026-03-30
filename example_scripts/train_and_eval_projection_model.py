import torch
from src.projection_model import PairwiseEmbeddingDataset, BalancedPairwiseEmbeddingDataset, ProjectionHead, train_projection_head, evaluate_auc


# Paths
train_pairs_csv = ''  # CSV containing labeled train pairs, having columns "variableId1", "variableId2", "similarity"
test_pairs_csv = ''  # CSV containing labeled test pairs, having columns "variableId1", "variableId2", "similarity"
emb_folders = ['']  # List of directory paths, containing the embeddings for the constructs in train_pairs_csv and test_pairs_csv
model_out_path = ''  # Output path to which the model is stored. Should have .pth suffix.

# Model and training settings
device = 'cpu'
input_dim = 4096 * len(emb_folders)  # 4096 is the embedding dimensionality
proj_dim = 2048
batch_size = 64
epochs = 30
lr = 1e-4
weight_decay = 0
margin = 0.1
delta = 0.05


if __name__ == '__main__':
    # initialize datasets
    train_data = BalancedPairwiseEmbeddingDataset(pairs_table=train_pairs_csv, emb_folders=emb_folders, neg_ratio=1.0, subset_fraction=0.2)
    test_data = PairwiseEmbeddingDataset(pairs_table=test_pairs_csv, emb_folders=emb_folders)

    # Initialize model
    model = ProjectionHead(input_dim=input_dim, proj_dim=proj_dim)

    # Train model
    model, _ = train_projection_head(
        model=model, 
        train_set=train_data, 
        val_set=None, 
        epochs_step_size=1, 
        batch_size=batch_size, 
        epochs=epochs, 
        lr=lr, 
        weight_decay=weight_decay, 
        margin=margin, 
        delta=delta, 
        device=device
    )

    if model_out_path is not None: 
        torch.save(model.state_dict(), model_out_path)

    # Evaluate model
    auc = evaluate_auc(
        model=model, 
        dataset=test_data, 
        device=device, 
        batch_size=256
    )

