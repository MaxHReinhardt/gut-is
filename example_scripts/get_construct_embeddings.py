import pandas as pd
from src.pretrained_text_embeddings import get_text_embeddings_huggingface, save_text_embeddings_from_dict


# Settings
model_name = 'nvidia/llama-embed-nemotron-8b'  # Name of the embedding model that shall be used for generating embeddings
names_descriptions_csv = ''  # File path of the CSV with construct information, having columns 'VariableId', 'GeneratedName', 'GeneratedDefinition'
output_folder = ''  # Target directory for storing the embeddings


if __name__ == '__main__':
    names_descriptions_df = pd.read_csv(names_descriptions_csv)

    # Embedding generation via huggingface
    embeddings_dict = get_text_embeddings_huggingface(
        names_descriptions_df=names_descriptions_df, 
        model_name=model_name, 
        include_names=True, 
        include_definitions=False, 
        batch_size=4
    )

    save_text_embeddings_from_dict(
        construct_embeddings=embeddings_dict, 
        output_folder=output_folder
    )

