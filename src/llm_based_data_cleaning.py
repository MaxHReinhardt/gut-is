import os.path
import xml.etree.ElementTree as ET
from openai import OpenAI
import csv
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import re


def get_text_from_txt(file_path):
    """
    Loads a txt file.

    Args:
        file_path: Path to txt file

    Returns:
        text as string
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def chunk_text_from_txt(text, chunk_size=600, overlap=200):
    """
    Chunks a single string.

    Args:
        text: Text string to be chunked
        chunk_size: chunk length in characters
        overlap: overlap between subsequent chunks

    Returns:
        list of strings
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
    return chunks


def get_text_and_tables_from_xml(path2xml):
    """
    Gets the text and tables from a grobid file.

    Args:
        path2xml: xml file path
    
    Returns:
        list with text strings
        list with table strings
    """

    if not path2xml.endswith('grobid.tei.xml'):
        return [], []

    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Parse the XML content from the file
    tree = ET.parse(path2xml)
    root = tree.getroot()

    # Lists to store extracted data
    text_data = []
    tables_data = []

    # Extract the text body
    for text_section in root.findall('.//tei:div', namespaces):
        for x in text_section:
            if x.tag == '{http://www.tei-c.org/ns/1.0}listBibl':
                continue
            text_data.append(' '.join(x.itertext()))

    # Extract the tables in XML formatting
    for table_section in root.findall('.//tei:figure', namespaces):
        if table_section.get('type') == 'table':
            table_xml = ET.tostring(table_section, encoding='unicode').replace('ns0:', '')
            end_header = table_xml.find('">')
            if end_header:
                table_xml = table_xml[(end_header + 2):]
                table_xml = table_xml.replace('</figure>', '')
            tables_data.append(table_xml)

    return text_data, tables_data


def chunk_text_from_xml(data, min_chunk_size=200, max_chunk_size=900, overlap=300):
    """
    Chunks the data from an XML file. Further, a minimum chunk size is ensured by concatenating subsequent strings
    until the desired size is reached. This mainly aims at concatenating headlines with subsequent paragraphs and
    preventing very short chunks due to wrongly formatted data.

    Args:
        data: List of strings
        min_chunk_size: minimal chunk size in characters
        max_chunk_size: maximum chunk size in characters
        overlap: Overlap between subsequent chunks, if they were formatted as a single string in the XML file

    Returns:
        List of chunked strings
    """

    # Ensure min chunk size (handled by concatenation with subsequent texts, as short elements are often headlines)
    acc_data = ""
    merged_data = []
    for string in data:
        acc_data += " " + string if acc_data else string  # Concat current text with accumulated texts
        # If accumulated text exceeds the threshold, add to output list
        if len(acc_data) >= min_chunk_size:
            merged_data.append(acc_data)
            acc_data = ""  # Reset accumulated text

    # Split texts further when their lengths exceed the threshold
    # As texts are not split based on content, chunks are decided to overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents(merged_data)
    chunks = [chunk.page_content for chunk in chunks]

    return chunks


def retrieve_chunks(queries, text_chunks, table_chunks, k, device, model=None):
    """
    Retrieve the top k chunks for a set of queries. The function receives two lists, one for text_chunks, one for
    table chunks. For each of these lists, the top k chunks and their scores are collected in separate outputs,
    but with scores on a shared scale. The length of text_chunks is expected to be larger than k.

    As sentence transformer model, "Alibaba-NLP/gte-Qwen2-1.5B-instruct" is used.
    Documentation: https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct

    Args:
        queries: list with query strings
        text_chunks: list with text chunks (must have more than k items)
        table_chunks: list with table chunks (might be empty)
        k: number of top chunks to return for each document list
        device: PyTorch device
        model: Allows to provide "Alibaba-NLP/gte-Qwen2-1.5B-instruct" (or another compatible model) pre-initialized to avoid multiple initializations.
            Can be set to None, in this case, "Alibaba-NLP/gte-Qwen2-1.5B-instruct" is loaded automatically with trust_remote_code=True.
    Returns:
        Two dicts of format {query_1: [(doc_1, score_doc_1), (doc_2, score_doc_2)], query_2: [...] } (one for
            texts, one for tables) if text_chunks has at least k items; None, None otherwise
    """

    # The number of text chunks should be >= k
    if len(text_chunks) < k:
        return None, None

    combined_chunks = text_chunks + table_chunks

    # Load the Sentence Transformer model only if not provided
    if model is None: 
        model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
        model.max_seq_length = 8192
        model.to(device)

    # Embed queries
    query_embeddings = model.encode(queries, prompt_name="query", convert_to_tensor=True)

    # Embed documents
    doc_embeddings = model.encode(combined_chunks, convert_to_tensor=True)

    # Compute similarities
    similarities = model.similarity(query_embeddings, doc_embeddings)
    topk_values, topk_indices = similarities.topk(len(combined_chunks))  # Return the values and indices of all chunks

    # Split the top k results back into text_chunks and table_chunks
    query_to_text_docs = {}
    query_to_table_docs = {}

    topk_indices = topk_indices.cpu()
    topk_values = topk_values.cpu()

    for query_idx in range(topk_indices.shape[0]):
        # Separate top-k results based on the chunk type
        text_docs_with_scores = []
        table_docs_with_scores = []

        for idx, score in zip(topk_indices[query_idx], topk_values[query_idx]):
            chunk = combined_chunks[idx]
            if (chunk in text_chunks) and (len(text_docs_with_scores) < k):
                text_docs_with_scores.append((chunk, float(score)))
            elif (chunk in table_chunks) and (len(table_docs_with_scores) < k):
                table_docs_with_scores.append((chunk, float(score)))
            elif (((len(text_docs_with_scores) >= k) and (len(table_docs_with_scores) >= k))
                  or ((len(text_docs_with_scores) >= k) and (len(table_docs_with_scores) >= len(table_chunks)))):
                break  # Break if both lists have desired size

        query_to_text_docs[queries[query_idx]] = text_docs_with_scores
        query_to_table_docs[queries[query_idx]] = table_docs_with_scores

    return query_to_text_docs, query_to_table_docs


def retrieve_chunks_for_constructs(construct_df, full_text_dir, file_format, k, max_table_chunks, device, model=None):
    """
    Extract the top k chunks for a collection of constructs from the full texts of their publications. If full texts
    are in XML format, a maximum number of table chunks can be specified. This is helpful for publications where
    construct names repeatedly appear in results tables. Still, in some publications, definitions might be provided
    in a table. Thus, a small value > 0 is recommended.

    Args:
        construct_df: Pandas df, expected to have columns 'name' (referring to construct name) and 'publication_id'
        full_text_dir: Directory containing the full text files (file names are expected to be the publication IDs)
        file_format: 'txt' or 'xml'
        k: number of top chunks to be retrieved for each construct
        max_table_chunks: maximum number of table chunks that are part of the top k chunks
        device: PyTorch device
        model: Allows to provide "Alibaba-NLP/gte-Qwen2-1.5B-instruct" (or another compatible model) pre-initialized to avoid multiple initializations.
            Can be set to None, in this case, "Alibaba-NLP/gte-Qwen2-1.5B-instruct" is loaded automatically with trust_remote_code=True.
    Returns
        Enriched construct_df with additional column 'full_text_chunks'
    """

    # Extract unique publication IDs
    unique_publication_ids = construct_df['publication_id'].unique()

    # Initialize an empty list to store results
    enriched_data = []

    # Iterate over publications
    for publication_id in unique_publication_ids:

        # Filter the df for the current publication_id
        publication_df = construct_df[construct_df['publication_id'] == publication_id]

        # Extract the list of unique construct names
        unique_construct_names = publication_df['name'].astype(str).unique().tolist()

        if file_format == 'txt':
            # Load and chunk full text of publication
            file_path = os.path.join(full_text_dir, str(publication_id) + '.pdf.txt')
            if os.path.exists(file_path):  # If full text file does exist
                full_text = get_text_from_txt(file_path)
                chunks = chunk_text_from_txt(text=full_text, chunk_size=900, overlap=300)

                # Extract relevant chunks for each construct
                doc_score_dict, _ = retrieve_chunks(
                    queries=unique_construct_names,
                    text_chunks=chunks,
                    table_chunks=[],
                    k=k,
                    device=device,
                    model=model
                )  # doc_score_dict has form {query_1: [(doc_1, score_doc_1), (doc_2, score_doc_2)], query_2: [...] }

                # Remove the score information to achieve form {query_1: [doc_1, doc_2], query_2: [...] }
                if doc_score_dict is not None:
                    context_dict = {query: [doc for doc, _ in doc_score_tuples]
                                    for query, doc_score_tuples in doc_score_dict}

                else:
                    # If less than k text chunks are available, set context_dict empty
                    context_dict = {}

            else:
                # If no full text is available, set context_dict empty
                context_dict = {}

        elif file_format == 'xml':
            # Load and chunk text and tables of publication
            file_path = os.path.join(full_text_dir, str(publication_id) + '.grobid.tei.xml')
            if os.path.exists(file_path):  # If full text file does exist
                text_data, tables_data = get_text_and_tables_from_xml(file_path)
                text_chunks = chunk_text_from_xml(text_data, min_chunk_size=200, max_chunk_size=900, overlap=300)
                table_chunks = chunk_text_from_xml(tables_data, min_chunk_size=200, max_chunk_size=900, overlap=300)

                # Extract relevant chunks for each construct
                text_doc_score_dict, table_doc_score_dict = retrieve_chunks(
                    queries=unique_construct_names,
                    text_chunks=text_chunks,
                    table_chunks=table_chunks,
                    k=k,
                    device=device,
                    model=model
                )

                if text_doc_score_dict is not None:
                    # Iterate over the constructs/queries
                    context_dict = {}
                    for query in unique_construct_names:
                        # Get the list of text and table docs with scores for the current query
                        text_docs_scores = text_doc_score_dict.get(query, [])
                        table_docs_scores = table_doc_score_dict.get(query, [])
                        # Sort text_docs and table_docs by score in descending order
                        text_docs_scores_sorted = sorted(text_docs_scores, key=lambda x: x[1], reverse=True)
                        table_docs_scores_sorted = sorted(table_docs_scores, key=lambda x: x[1], reverse=True)

                        # Collect the top_k docs while respecting the specified max_table_chunks
                        final_docs = []
                        text_idx, table_idx = 0, 0
                        while len(final_docs) < k:
                            # If the table_doc_score_dict has sufficient length, max_table_chunks is not reached and
                            #  table chunk has a higher score than text chunk, append table chunk
                            if ((table_idx < len(table_docs_scores_sorted))
                                    and (table_idx < max_table_chunks)
                                    and table_docs_scores_sorted[table_idx][1] > text_docs_scores_sorted[text_idx][1]):
                                final_docs.append(table_docs_scores_sorted[table_idx][0])
                                table_idx += 1

                            # Else append text chunk
                            else:
                                final_docs.append(text_docs_scores_sorted[text_idx][0])
                                text_idx += 1

                        # Store the final docs for the current query
                        context_dict[query] = final_docs

                else:
                    # If less than k text chunks are available, set context_dict empty
                    context_dict = {}

            else:
                # If no full text is available, set context_dict empty
                context_dict = {}

        else:
            raise ValueError("file_format must be 'txt' or 'xml'.")

        # Iterate over constructs of publication
        for _, row in publication_df.iterrows():
            # Extract context for the construct or leave empty if no full text
            context_list = context_dict.get(row['name'], [])
            combined_context = "\n---\n".join(context_list)

            # Create enriched row
            enriched_row = row.to_dict()
            enriched_row['full_text_chunks'] = combined_context
            enriched_data.append(enriched_row)

        print(f"Done with publication {publication_id}.")

    # Convert enriched data back to a df
    enriched_df = pd.DataFrame(enriched_data)

    return enriched_df


def rag_names_and_definitions_for_constructs(construct_df_with_chunks, output_file_path, credentials, generate_names=True, generate_definitions=True):
    """
    We used this function for the RAG of names and definitions on DISKNET.
    Generates construct names and definitions using pre-retrieved full text chunks, if provided.
    As chat completion model, "llama-3.3-70b-instruct" is used via SAIA (https://docs.hpc.gwdg.de/services/ai-services/saia/index.html).
    Results are directly written to CSV.

    Args:
        construct_df_with_chunks: Enriched disknet_construct_df with additional column 'full_text_chunks' (as output from retrieve_chunks_for_constructs)
        output_file_path: Path of empty output csv (will be created if not existent)
        credentials: SAIA API key
        generate_names: Boolean, indicating if construct names should be generated
        generate_definitions: Boolean, indicating if construct definitions should be generated
    """

    # Create output csv, if it does not exist
    column_names = ['id', 'original_name', 'generated_name', 'definition', 'generated_definition', 'publication_id', 'publication_name', 'full_text_chunks']
    file_exists = os.path.isfile(output_file_path)  # Check if the file exists
    with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(output_file_path).st_size == 0:
            writer.writerow(column_names)

    # Initialize client and model
    client = OpenAI(
        api_key=credentials,
        base_url="https://chat-ai.academiccloud.de/v1"
    )
    model = "llama-3.3-70b-instruct"

    # Iterate over constructs
    for i, row in construct_df_with_chunks.iterrows():

        # Construct the base prompt
        base_prompt_elements = []

        if pd.isna(row['name']):
            continue  # Omit construct if name is missing
        else:
            base_prompt_elements.append(f'Construct name: {row["name"]}.')

        if pd.isna(row['definition']):
            pass
        else:
            base_prompt_elements.append(f'Construct definition: {row["definition"]}.')

        if pd.isna(row['publication_name']):
            pass
        else:
            base_prompt_elements.append(f'Scientific publication the construct was used in: {row["publication_name"]}.')

        if pd.isna(row['full_text_chunks']):
            pass
        else:
            base_prompt_elements.append(f'Relevant chunks from publication full text: {row["full_text_chunks"]}.')

        # Generate construct name, if requested
        if generate_names:
            # Construct the name prompt
            name_prompt_elements = base_prompt_elements.copy()
            name_prompt_elements.append(
                'Based on the given information, provide the cleaned construct name without any additional words or '
                'phrases.'
            )
            name_prompt = " ".join(name_prompt_elements)

            name_system_msg = (
                """
                You are an automated system for a data cleaning task. Your objective is to provide a cleaned version of 
                the name of a construct that was used in a scientific publication as part of a statistical model.
                You are provided with the construct name, construct definition, publication name, and relevant excerpts 
                from the full text of the publication, as far as these information are available.

                Guidelines:
                1. Ensure the construct name is in full English words.
                2. Expand abbreviations only if they add meaningful context to the name.
                3. Exclude abbreviations that merely repeat parts of the construct name.
                4. Correct any spelling or grammatical errors in the construct name.
                5. Do not make any other changes to the construct name.

                Your response should only include the construct definition, without any additional words or
                phrases.
                """
            )

            # Get generated name
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": name_system_msg},
                          {"role": "user", "content": name_prompt}],
                model=model,
                temperature=0.3  # Low temperature as data cleaning requires more consistency than creativity
            )

            completion = chat_completion.choices[0].message.content

            # If completion contains a chain of thought (e.g. for DeepSeek models), remove it
            match_after_thought = re.search(r"</think>\s*\n*(.+)", completion, re.DOTALL)
            generated_name = match_after_thought.group(1).strip() if match_after_thought else completion.strip()

        else:
            generated_name = ''

        # Generate construct definitions, if requested
        if generate_definitions:
            # Construct the definition prompt
            description_prompt_elements = base_prompt_elements.copy()
            description_prompt_elements.append(
                'Based on the provided information, generate a construct definition.'
            )
            description_prompt = " ".join(description_prompt_elements)

            # Define the definition system message
            description_system_msg = (
                """
                You are an automated system for the context-aware generation of construct definitions. All constructs 
                were used in scientific publications as part of a statistical model. You are provided with the construct 
                name, construct definition, publication name, and relevant excerpts from the full text of the 
                publication, as far as these information are available. Your task is to generate a meaningful but 
                concise definition of the construct, based on the given information.

                If an explicit construct definition is provided or can be derived from the provided excerpts of the
                publication text, clean and refine the original definition. This includes, 
                1. expanding abbreviations into their full English forms, 
                2. correcting any spelling or grammatical errors,
                3. ensuring the definition is meaningful and concise,
                4. staying as close as possible to the provided definition.

                Your response should only include the construct definition, without any additional words or
                phrases.
                """
            )

            # Get generated definition
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": description_system_msg},
                          {"role": "user", "content": description_prompt}],
                model=model,
                temperature=0.3  # Low temperature as data cleaning requires more consistency than creativity
            )

            completion = chat_completion.choices[0].message.content

            # If completion contains a chain of thought (e.g. for DeepSeek models), remove it
            match_after_thought = re.search(r"</think>\s*\n*(.+)", completion, re.DOTALL)
            generated_description = match_after_thought.group(1).strip() if match_after_thought else completion.strip()

        else:
            generated_description = ''

        # Create a new row in the output CSV
        new_row = [
            row['id'] if not pd.isna(row['id']) else '',
            row['name'] if not pd.isna(row['name']) else '',
            generated_name,
            row['definition'] if not pd.isna(row['definition']) else '',
            generated_description,
            row['publication_id'] if not pd.isna(row['publication_id']) else '',
            row['publication_name'] if not pd.isna(row['publication_name']) else '',
            row['full_text_chunks'] if not pd.isna(row['full_text_chunks']) else ''
        ]
        with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

        print(f"Processed construct {i} with name {row['name']}.")

        # Wait for 1.2 seconds after each construct to ensure staying within LLM usage limits
        time.sleep(1.2)


def generate_names_and_multiple_definitions_for_constructs(construct_df, output_file_path, model, credentials, generate_names=True, generate_definitions=True, 
                                                           num_definition_versions=1):
    """
    We used this function for the generation of names and definitions on the gold standard dataset from Larsen & Bong.
    If desired, data is augmented by generating (multiple) paraphrased definitions.
    Embdeddings are generated via the SAIA API (https://docs.hpc.gwdg.de/services/ai-services/saia/index.html).
    Results are directly written to CSV.

    Args:
        construct_df: df with construct information
        output_file_path: Path of empty output csv (will be created if not existent)
        model: Name of the SAIA embedding model of choice (e.g. e5-mistral-7b-instruct)
        credentials: SAIA API key
        generate_names: Boolean, indicating if construct names should be generated
        generate_definitions: Boolean, indicating if construct definitions should be generated
        num_definition_versions: Total number of definitions to generate per construct
    """

    # Create output csv, if it does not exist
    column_names = ['VariableId', 'VariableName', 'VariableDefinition', 'GeneratedName', 'GeneratedDefinition', 'ItemTexts']
    file_exists = os.path.isfile(output_file_path)  # Check if the file exists
    with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(output_file_path).st_size == 0:
            writer.writerow(column_names)

    # Initialize client and model
    client = OpenAI(
        api_key=credentials,
        base_url="https://chat-ai.academiccloud.de/v1"
    )

    # Iterate over constructs
    for i, row in construct_df.iterrows():

        # Construct the base prompt
        base_prompt_elements = []

        if pd.isna(row['VariableName']):
            continue  # Omit construct if name is missing
        else:
            base_prompt_elements.append(f'Construct name: {row["VariableName"]}.')

        if pd.isna(row['VariableDefinition']):
            pass
        else:
            base_prompt_elements.append(f'Construct definition: {row["VariableDefinition"]}.')

        if pd.isna(row['ItemTexts']):
            pass
        else:
            base_prompt_elements.append(f'Texts of the items used to measure the construct: {row["ItemTexts"]}.')

        # Generate construct name, if requested
        if generate_names:
            # Construct the name prompt
            name_prompt_elements = base_prompt_elements.copy()
            name_prompt_elements.append('Based on the given information, provide the cleaned construct name without any additional words or phrases.')
            name_prompt = " ".join(name_prompt_elements)

            name_system_msg = (
                """
                You are an automated system for a data cleaning task. Your objective is to provide a cleaned version of 
                the name of a construct that was used in a scientific publication as part of a statistical model.
                You are provided with the construct name, construct definition, and the texts of the items used to measure the construct, 
                as far as these information are available.

                Guidelines:
                1. Ensure the construct name is in full English words.
                2. Expand abbreviations only if they add meaningful context to the name.
                3. Exclude abbreviations that merely repeat parts of the construct name.
                4. Correct any spelling or grammatical errors in the construct name.
                5. Do not make any other changes to the construct name.

                Your response should only include the construct definition, without any additional words or
                phrases.
                """
            )

            # Get generated name
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": name_system_msg},
                          {"role": "user", "content": name_prompt}],
                model=model,
                temperature=0.3  # Low temperature as data cleaning requires more consistency than creativity
            )

            completion = chat_completion.choices[0].message.content

            # If completion contains a chain of thought (e.g. for DeepSeek models), remove it
            match_after_thought = re.search(r"</think>\s*\n*(.+)", completion, re.DOTALL)
            generated_name = match_after_thought.group(1).strip() if match_after_thought else completion.strip()

        else:
            generated_name = ''

        # Generate construct definitions, if requested
        if generate_definitions:
            # Construct the definition prompt
            description_prompt_elements = base_prompt_elements.copy()
            description_prompt_elements.append('Based on the given information, provide the generated or refined construct definition without any additional words or phrases.')
            description_prompt = " ".join(description_prompt_elements)

            # Define the definition system message
            description_system_msg = (
                """
                You are an automated system for the context-aware generation of construct definitions. All constructs 
                were used in scientific publications as part of a statistical model. You are provided with the construct name, 
                construct definition, and the texts of the items used to measure the construct, 
                as far as these information are available. Your task is to generate a meaningful but 
                concise definition of the construct, based on the given information.

                If an explicit construct definition is provided, clean and refine the original definition. This includes, 
                1. expanding abbreviations into their full English forms, 
                2. correcting any spelling or grammatical errors,
                3. ensuring the definition is meaningful and concise,
                4. staying as close as possible to the provided definition.

                Your response should only include the construct definition, without any additional words or
                phrases.
                """
            )

            # Get generated definition
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": description_system_msg},
                        {"role": "user", "content": description_prompt}],
                model=model,
                temperature=0.3  # Low temperature as data cleaning requires more consistency than creativity
            )

            completion = chat_completion.choices[0].message.content

            # If completion contains a chain of thought (e.g. for DeepSeek models), remove it
            match_after_thought = re.search(r"</think>\s*\n*(.+)", completion, re.DOTALL)
            generated_description = match_after_thought.group(1).strip() if match_after_thought else completion.strip()

            # Create a new row in the output CSV
            new_row = [
                f"{row['VariableId'] if not pd.isna(row['VariableId']) else ''}_{0}",
                row['VariableName'] if not pd.isna(row['VariableName']) else '',
                row['VariableDefinition'] if not pd.isna(row['VariableDefinition']) else '',
                generated_name,
                generated_description,
                row['ItemTexts'] if not pd.isna(row['ItemTexts']) else ''
            ]
            with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(new_row)

            print(f"Processed construct {i} with name {row['VariableName']}.")

            # Wait after each construct to ensure staying within LLM usage limits
            time.sleep(1.0)

            # Generate the required amount of paraphrased definition versions per construct 
            for j in range(1, num_definition_versions):
                description_prompt_elements = []
                description_prompt_elements.append((f'Construct name: {generated_name}.'))
                description_prompt_elements.append(f'Construct definition: {generated_description}.')
                description_prompt_elements.append('Provide the rephrased construct definition without any additional words or phrases.')

                description_system_msg = (
                """
                You are a data augmentation system for the generation of rephrased construct definitions. All constructs 
                were used in scientific publications as part of a statistical model. You are provided with the construct name 
                and definition. Based on these information, generate a rephrased version of the construct definition.

                Guidelines:
                1. Vary the wording of the definition significantly.
                2. Preserve the meaning of the definition.
                3. Keep the definition as concise as possible.

                Your response should only include the rephrased definition, without any additional words or phrases.
                """
                )
                description_prompt = " ".join(description_prompt_elements)

                # Get generated definition
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "system", "content": description_system_msg},
                            {"role": "user", "content": description_prompt}],
                    model=model,
                    temperature=0.3  # Low temperature as data cleaning requires more consistency than creativity
                )

                completion = chat_completion.choices[0].message.content

                # If completion contains a chain of thought (e.g. for DeepSeek models), remove it
                match_after_thought = re.search(r"</think>\s*\n*(.+)", completion, re.DOTALL)
                generated_description = match_after_thought.group(1).strip() if match_after_thought else completion.strip()

                # Create a new row in the output CSV
                new_row = [
                    f"{row['VariableId'] if not pd.isna(row['VariableId']) else ''}_{j}",
                    row['VariableName'] if not pd.isna(row['VariableName']) else '',
                    row['VariableDefinition'] if not pd.isna(row['VariableDefinition']) else '',
                    generated_name,
                    generated_description,
                    row['ItemTexts'] if not pd.isna(row['ItemTexts']) else ''
                ]
                with open(output_file_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row)

                # Wait after each construct to ensure staying within LLM usage limits
                time.sleep(1.0)

