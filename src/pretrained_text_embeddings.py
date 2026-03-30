import os
import time
import pandas as pd
import numpy as np
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer


def construct_string_for_text_embedding(construct_name, construct_definition, construct_relations,
                                        include_name=True, include_definition=True, include_relations=True):
    """
    Constructs a string as the basis of a text embedding of a construct, containing selected components.
    """

    embedding_string = ""

    if include_name:
        embedding_string += f"Construct: {construct_name}"

    if include_definition:
        embedding_string += f"Definition: {construct_definition}"

    if include_relations:
        embedding_string += f"Construct relations: {construct_relations}"

    embedding_string = embedding_string.strip()  # Remove trailing newline characters, if any

    return embedding_string
    

def get_text_embeddings_saia(names_descriptions_df, model, credentials, include_names=True, include_definitions=True):
    """
    Gets construct embeddings from the SAIA API (https://docs.hpc.gwdg.de/services/ai-services/saia/index.html).

    Args:
        names_descriptions_df: Df where each row corresponds to one construct. Expected to have columns 'VariableId', 'GeneratedName' 
            (containing the cleaned construct name), and 'GeneratedDefinition' (containing the cleaned construct definition)
        model: Name of the SAIA model of choice (e.g. e5-mistral-7b-instruct)
        credentials: SAIA API key
        include_names: Indicates if construct names should be included in the embedding string 
        include_definitions: Indicates if construct definitions should be included in the embedding string 
    
    Returns:
        A dictionary mapping 'VariableId' to embedding
    """

    # Initialize client
    client = OpenAI(
        api_key=credentials,
        base_url="https://chat-ai.academiccloud.de/v1"
    )

    # Initialize results dict
    construct_embeddings = {}

    # Iterate over constructs
    for i, row in names_descriptions_df.iterrows():
        # Prepare strings for construct
        construct_name = row['GeneratedName'] if include_names else None
        construct_definition = row['GeneratedDefinition'] if include_definitions else None

        # Construct embedding string
        embedding_string = construct_string_for_text_embedding(
            construct_name=construct_name,
            construct_definition=construct_definition,
            construct_relations=None,
            include_name=include_names,
            include_definition=include_definitions,
            include_relations=False
        )

        # Generate text embedding
        response = client.embeddings.create(
            input=[embedding_string],
            model=model
        )
        embedding = response.data[0].embedding

        # Add embedding for construct to results dict
        construct_embeddings[row['VariableId']] = embedding
        print(f"Done with {i}.")

        # Wait after each construct to ensure staying within LLM usage limits
        time.sleep(1.0)

    return construct_embeddings
    

def get_text_embeddings_huggingface(names_descriptions_df, model_name, include_names=True, include_definitions=True, batch_size=32):
    """
    Generates construct embeddings via Huggingface.

    Args:
        names_descriptions_df: Df where each row corresponds to one construct. Expected to have columns 'VariableId', 'GeneratedName' 
            (containing the cleaned construct name), and 'GeneratedDefinition' (containing the cleaned construct definition)
        model_name: Name of the embedding model of choice. "Qwen/Qwen3-Embedding-8B", "nvidia/llama-embed-nemotron-8b", and "intfloat/e5-mistral-7b-instruct" are 
            implemented. Note that "intfloat/e5-mistral-7b-instruct" was used via SAIA for the experiments, results obtained using Huggingface might deviate.
            Also note that "nvidia/llama-embed-nemotron-8b" is loaded with trust_remote_code=True.
        include_names: Indicates if construct names should be included in the embedding string 
        include_definitions: Indicates if construct definitions should be included in the embedding string 
        batch_size: Size of batches with which embeddings are processed
    
    Returns:
        A dictionary mapping 'VariableId' to embedding
    """

    # Load model
    if model_name == "Qwen/Qwen3-Embedding-8B":
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

    elif model_name == "nvidia/llama-embed-nemotron-8b":
        attn_implementation = "eager"
        model = SentenceTransformer(
            "nvidia/llama-embed-nemotron-8b",
            trust_remote_code=True,
            model_kwargs={"attn_implementation": attn_implementation, "torch_dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left"},
        )

    elif model_name == "intfloat/e5-mistral-7b-instruct":
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")

    else: 
        raise ValueError('model_name is invalid or not available.')

    construct_embeddings = {}

    texts = []
    ids = []

    # Build all input strings first for batching
    for _, row in names_descriptions_df.iterrows():
        construct_name = row['GeneratedName'] if include_names else None
        construct_definition = row['GeneratedDefinition'] if include_definitions else None

        embedding_string = construct_string_for_text_embedding(
            construct_name=construct_name,
            construct_definition=construct_definition,
            construct_relations=None,
            include_name=include_names,
            include_definition=include_definitions,
            include_relations=False
        )

        texts.append(embedding_string)
        ids.append(row['VariableId'])

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Map back to dict
    for var_id, emb in zip(ids, embeddings):
        construct_embeddings[var_id] = emb

    return construct_embeddings


def save_text_embeddings_from_dict(construct_embeddings, output_folder):
    """
    Save generated embeddings.

    Args: 
        construct_embeddings: A dictionary mapping 'VariableId' to embedding
        output_folder: Target directory
    """

    os.makedirs(output_folder, exist_ok=True)

    for construct_id, embedding in construct_embeddings.items():
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        output_path = os.path.join(output_folder, f"{construct_id}.pt")
        torch.save(embedding_tensor, output_path)
        print(f"Saved {construct_id}.pt")

    print(f"All embeddings saved to {output_folder}")

