# GUT-IS: A Data-Driven Approach to Integrating Constructs and Their Relations in Information Systems
This repository contains the source code of the paper "A Data-Driven Approach to Integrating Constructs and Their Relations in Information Systems" (submitted to ECIS 2026).


## 1 Project overview
While structural equation models are widely used in IS research, the inconsistent use of constructs impedes cumulative knowledge development. This project introduces GUT-IS, a scalable workflow that aims at unifying multiple SEM models by consolidating their constructs while reconciling their relations. Therefore, we calculate pairwise similarities between constructs and subsequently apply clustering algorithms to the resulting similarity graph which yields globally coherent construct groupings. Optimal cluster solutions are selected using a loss function that balances semantic purity and parsimony in the number of clusters via a weighting parameter $\alpha$. We conduct experiments based on the labeled dataset from Larsen & Bong (2016), and on the DISKNET data (Dann et al., 2019).


## 2 Prerequisites
- This project was developed using Python 3.12.10. 
- The requirements can be installed via `pip install -r requirements.txt`.
- Parts of our workflow use LLMs via the SAIA API, which requires an API key. More information can be found on: https://docs.hpc.gwdg.de/services/ai-services/saia/index.html


## 3 Workflow and repository structure
The repository structure follows our workflow and consists of the following steps: (1) data cleaning and enrichment using LLMs, (2) representing constructs using pretrained text embeddings, (3) training and evaluating a task-specific projection model on top of the pretrained embeddings, and (4) clustering constructs while balancing the parsimony and purity of the resulting partition. For each of these steps, we provide the source code (in `/src`) and exemplary scripts (in `/example_scripts`). 

As the utilized data is not fully public and includes the full texts of publications, our repository cannot provide an end-to-end reproduction of our experiments. Instead, the example scripts (in `/example_scripts`) are intended to serve as an entry point for applying our workflow to custom data. 

In addition, to allow for a manual exploration of our approach, we provide exemplary data in form of the DISKNET constructs, along with precomputed similarity relationships and cluster solutions, as well as a Jupyter notebook that displays cluster solutions across $\alpha$-values (in `/DISKNET_analysis`).

### 3.1 Data cleaning and enrichment
**Source code:** `/src/llm_based_data_cleaning.py`  
**Example script:** `/example_scripts/generate_names_definitions.py`

Construct names and definitions are cleaned/generated using LLMs, based on meaningful context. On DISKNET (Dann et al., 2019), we employ a RAG approach and retrieve chunks from the full texts of publications as context. For the dataset from Larsen & Bong (2016), we utilize construct items.

### 3.2 Representing constructs using pretrained text embeddings
**Source code:** `/src/pretrained_text_embeddings.py`  
**Example script:** `/example_scripts/get_construct_embeddings.py`

The cleaned construct names and definitions are embedded using SOTA pretrained models via Huggingface and SAIA.

### 3.3 Training and evaluating a task-specific projection model
**Source code:** `/src/projection_model.py`  
**Example script:** `/example_scripts/train_and_eval_projection_model.py`

The pretrained embeddings are projected in a task-specific space using a lightweight projection model. Due to its low parameter count, the model can be trained with the limited labels in the Larsen & Bong (2016) dataset.

### 3.4 Clustering constructs while balancing parsimony and purity
**Source code:** `/src/clustering.py`, `/src/parsimony_purity_trade_off.py`  
**Example script:** `/example_scripts/explore_parsimony_purity_continuum.py`  
**Qualitative exploration:** `/DISKNET_analysis/analyse_DISKNET_clusters.ipynb`

At first, cluster solutions are computed across a grid of algorithmic configurations. Following, the parsimony and purity losses for each configuration are calculated which allows to identify the optimal cluster solution for each weighted combination of these losses (controlled by weighting parameter $\alpha$).


## 4 References
Dann, D., Maedche, A., Teubner, T., Mueller, B., Meske, C., & Funk, B. (2019). DISKNET - A Platform for the Systematic Accumulation of Knowledge in IS Research. ICIS 2019 Proceedings.

Larsen, K. R., & Bong, C. H. (2016). A Tool for Addressing Construct Identity in Literature Reviews and Meta-Analyses. MIS Quarterly, 40(3), 529-551.
