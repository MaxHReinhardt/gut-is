from src.llm_based_data_cleaning import retrieve_chunks_for_constructs, rag_names_and_definitions_for_constructs, generate_names_and_multiple_definitions_for_constructs
import pandas as pd
from sentence_transformers import SentenceTransformer


# Paths
construct_df_path = ''  # CSV with construct information
full_text_dir = ''  # directory containing the full texts of publications, only required for RAG
out_path = ''  # Output CSV path 

# Credentials
saia_api_key = ''

# Settings
device = 'cpu'
generate_names = True
generate_definitions = True
num_definition_versions = 1  # Number of definition versions to generate when using generate_names_and_multiple_definitions_for_constructs


if __name__ == '__main__':
    # Load the construct data
    construct_df = pd.read_csv(construct_df_path)
    construct_df = construct_df[:5]

    #
    ## Retrieval augmented generation of names and definitions (for DISKNET)
    #

    # Initialize the embedding model
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  # In the experiments, we used "Alibaba-NLP/gte-Qwen2-1.5B-instruct", here, we switched to a lighter model with less requirements as example
    # embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)  # Note that trust_remote_code=True
    # embedding_model.max_seq_length = 8192
    # embedding_model.to(device)

    # Retrieve context from full texts
    construct_df_with_chunks = retrieve_chunks_for_constructs(
        construct_df=construct_df,
        full_text_dir=full_text_dir,
        file_format='xml',
        k=4,
        max_table_chunks=2,
        device=device,
        model=embedding_model
    )

    # Generate names and definitions 
    rag_names_and_definitions_for_constructs(
        construct_df_with_chunks=construct_df_with_chunks,
        output_file_path=out_path,
        credentials=saia_api_key,
        generate_names=generate_names,
        generate_definitions=generate_definitions
    )

    #
    ## Generation of names and multiple definitions (for Larsen & Bong gold label dataset)
    #

    # generate_names_and_multiple_definitions_for_constructs(
    #     construct_df=construct_df, 
    #     output_file_path=out_path, 
    #     model='llama-3.3-70b-instruct', 
    #     credentials=saia_api_key, 
    #     generate_names=generate_names, 
    #     generate_definitions=generate_definitions, 
    #     num_definition_versions=num_definition_versions
    # )

