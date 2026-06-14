import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import webbrowser


def analyze_clusters(cluster_df, construct_df, k=5, search_construct=None):
    """
    Allows to display a cluster solution for manual analysis.

    Args:
        cluster_df: df assigning constructs to clusters, expected to have columns 'construct_id' and 'cluster_id' 
        construct_df: df with construct information, expected to have columns 'id' and 'generated_name'
        k: Top k clusters to display
        search_construct: Construct to search for. If specified, only the cluster containing the construct is displayed.
    """

    # Merge construct and cluster data
    merged_df = pd.merge(
        construct_df[['id', 'generated_name']],
        cluster_df,
        left_on='id',
        right_on='construct_id'
    ).drop(columns=['id'])

    # Count clusters and constructs
    num_clusters = merged_df['cluster_id'].nunique()
    num_constructs = merged_df['construct_id'].nunique()
    print(f"\nTotal number of clusters: {num_clusters}")
    print(f"\nTotal number of constructs: {num_constructs}")

    # Identify top clusters
    cluster_sizes = merged_df['cluster_id'].value_counts()
    # print("Top clusters by number of constructs:")
    # print(cluster_sizes.head(k))

    # If search_construct is specified, find the cluster containing it
    if search_construct is not None:
        search_lower = search_construct.lower()
        matched_rows = merged_df[merged_df['generated_name'].str.lower() == search_lower]
        if matched_rows.empty:
            print(f"\nNo cluster contains the construct '{search_construct}'.")
            return
        cluster_to_show = matched_rows['cluster_id'].iloc[0]
        clusters_to_iterate = [cluster_to_show]
        print(f"\nShowing only cluster {cluster_to_show} containing '{search_construct}':")
    else:
        clusters_to_iterate = cluster_sizes.head(k).index

    # Show construct names and counts for selected clusters
    for cluster in clusters_to_iterate:
        cluster_rows = merged_df[merged_df['cluster_id'] == cluster].copy()

        # Group by lowercase generated_name for case-insensitive lexical matching
        grouped = (
            cluster_rows.groupby(cluster_rows['generated_name'].str.lower())
            .agg(
                occurrences=('generated_name', 'size'),
                generated_name=('generated_name', 'first')
            )
            .reset_index(drop=True)
            .sort_values(by='occurrences', ascending=False)
        )

        # Print constructs with occurrences
        print(f"\nCluster {cluster} (size: {cluster_sizes[cluster]}):")
        for _, row in grouped.iterrows():
            print(f"  - {row['generated_name']} [occurrences: {row['occurrences']}]")


def plot_global_model(cluster_df, construct_df, relation_df, min_relation_votes=1, min_cluster_size=1, search_construct=None, neighbor_deg=1, 
                      interactive=False, figsize=(12, 10), html_file=None):
    """
    Allows displaying the global model for a cluster solution. Cluster names and the signs of inter-cluster relationships are determined via majority voting.

    Args:
        cluster_df: df assigning constructs to clusters, expected to have columns 'construct_id' and 'cluster_id' 
        construct_df: df with construct information, expected to have columns 'id' and 'generated_name'
        relation_df: df with relation information, expected to have columns 'construct_id_1', 'construct_id_2', and 'path_coefficient'
        min_relation_votes: minimum number of relations from one cluster to another
        min_cluster_size: minimum number of constructs in a cluster
        search_construct: Only the clusters containing the specified construct (focal clusters) and their neighbors are displayed. Set None for not applying this filter.
        neighbor_deg: maximal degree of neighbors of focal clusters
        interactive: If True, an interactive plot is produced and opened in the Browser (recommended for larger graphs). If False, a static plot is produced.
        figsize: size of the static plot
        html_file: path to store the .html file of the interactive plot
    """

    #
    # Construct construct/cluster mapping and filter cluster by size
    # 

    # Merge construct and cluster data
    construct_cluster_df = pd.merge(
        construct_df[['id', 'generated_name']],
        cluster_df,
        left_on='id',
        right_on='construct_id'
    ).drop(columns=['id'])

    # ensure cluster IDs are ints
    construct_cluster_df["cluster_id"] = (construct_cluster_df["cluster_id"].astype(int))

    # Determine cluster sizes
    cluster_sizes = (construct_cluster_df["cluster_id"].value_counts().sort_index())

    # Filter on the clusters bigger than min_cluster_size
    valid_clusters = set(cluster_sizes[cluster_sizes >= min_cluster_size].index)
    construct_cluster_df = construct_cluster_df[construct_cluster_df["cluster_id"].isin(valid_clusters)].copy()

    if construct_cluster_df.empty:
        print("No clusters remain after filtering.")
        return

    # 
    # Determine cluster names by majority voting
    # 

    def majority_vote_name(group):
        lower_names = (group["generated_name"].str.lower())
        winner = (lower_names.value_counts().idxmax())
        winner_with_casing = group.loc[lower_names == winner, "generated_name"].iloc[0]
        return winner_with_casing

    cluster_name_map = construct_cluster_df.groupby("cluster_id").apply(majority_vote_name).to_dict()

    #
    # Collect inter-cluster relations
    # 

    # Initialize collection of the relations between clusters
    relation_rows = []

    # Construct a construct to cluster mapping dict for convenience
    construct_cluster_map = dict(zip(construct_cluster_df["construct_id"], construct_cluster_df["cluster_id"]))

    for r in relation_df.itertuples(index=False):
        c1 = r.construct_id_1
        c2 = r.construct_id_2

        # Ignore relations involcing constructs that were filtered out
        if (c1 not in construct_cluster_map or c2 not in construct_cluster_map):
            continue
        
        # Determine source and target cluster
        src_cluster = int(construct_cluster_map[c1])
        tgt_cluster = int(construct_cluster_map[c2])

        # Ignore relations between constructs in the same cluster
        if src_cluster == tgt_cluster:
            continue
        
        # Collect relation
        relation_rows.append({
            "source_cluster": src_cluster,
            "target_cluster": tgt_cluster,
            "sign": "positive" if r.path_coefficient > 0 else "negative"
        })

    if not relation_rows:
        print("No inter-cluster relations found.")
        return

    relation_rows = pd.DataFrame(relation_rows)

    # 
    # Aggregate inter-cluster relations
    # 

    aggregated = []

    for (src, tgt), group in relation_rows.groupby(["source_cluster", "target_cluster"]):
        # Count votes
        positive_votes = int((group["sign"] == "positive").sum())
        negative_votes = int((group["sign"] == "negative").sum())
        total_votes = (positive_votes + negative_votes)

        # Filter out inter-cluster relations with insufficient votes
        if total_votes < min_relation_votes:
            continue
        
        # Determine dominant relation sign
        if positive_votes >= negative_votes:
            dominant_sign = "positive"
            support = positive_votes
        else:
            dominant_sign = "negative"
            support = negative_votes

        # Collect relation and summary stats
        aggregated.append({
            "source_cluster": int(src),
            "target_cluster": int(tgt),
            "source_name": cluster_name_map[int(src)],
            "target_name": cluster_name_map[int(tgt)],
            "relation": dominant_sign,
            "support": support,
            "positive_votes": positive_votes,
            "negative_votes": negative_votes,
            "total_votes": total_votes,
            "consensus":support / total_votes
        })

    meta_relation_df = pd.DataFrame(aggregated)

    if meta_relation_df.empty:
        print("No cluster relations remain.")
        return

    #
    # Filter on clusters containing the selected search_construct, and their neighbors, if specified
    #

    if search_construct is not None:

        # Search for constructs matching the specified name
        matches = construct_cluster_df[construct_cluster_df["generated_name"].str.lower().eq(search_construct.lower())]

        if matches.empty:
            raise ValueError(f"Construct '{search_construct}' not found.")

        # Select clusters containing the matching constructs as focal clusters
        focal_clusters = set(matches["cluster_id"].astype(int).unique().tolist())

        # Construct a networkX graph based in the collected inter-cluster relations
        cluster_graph = nx.Graph()
        for r in meta_relation_df.itertuples():
            cluster_graph.add_edge(
                int(r.source_cluster),
                int(r.target_cluster)
            )

        # Initialize a collection of visible clusters
        visible_clusters = set()

        for focal_cluster in focal_clusters:
            
            # If the focal cluster has no relations, add it to visible cluster but don't search for neighbors
            if focal_cluster not in cluster_graph:
                visible_clusters.add(focal_cluster)
                continue
            
            # If the focal cluster has relations, add all clusters in a sufficiently small neighborhood to visible clusters
            neighborhood = nx.single_source_shortest_path_length(cluster_graph, focal_cluster, cutoff=neighbor_deg)
            visible_clusters.update(neighborhood.keys())

        # Filter meta_relation_df on the relations between visible clusters
        meta_relation_df = meta_relation_df[
            meta_relation_df["source_cluster"].isin(visible_clusters) & meta_relation_df["target_cluster"].isin(visible_clusters)
        ].copy()
    
    else:
        # If now construct name is specified, all clusters are visible with no cluster being a focal cluster
        visible_clusters = set(cluster_name_map.keys())
        focal_clusters = set()

    #
    # Interactive visualization
    #

    if interactive:
        if html_file == None:
            raise ValueError('Generating the interactive visualization requires specifying a path for storing the generated .html file.')
        
        try:
            from pyvis.network import Network
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "The pyvis library is required for interactive=True."
            )

        # Initialize network and its core settings
        net = Network(
            height="900px",
            width="100%",
            directed=True,
            notebook=True,
            bgcolor="white",
            cdn_resources="in_line"
        )

        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.1,
            spring_length=180,
            spring_strength=0.01
        )

        net.options.physics.enabled = True
        
        net.options.interaction = {
            "hover": True,
            "keyboard": True
        }

        for cluster_id in visible_clusters:
            cluster_size = int(cluster_sizes.loc[cluster_id])
            color = ("#ffcc00" if cluster_id in focal_clusters else "#97c2fc")

            net.add_node(
                cluster_id,
                label=str(cluster_name_map[cluster_id]),
                color=color,
                size=15 + cluster_size,
                title=f"""
                Name: {cluster_name_map[cluster_id]} (determined via majority voting)
                Cluster ID: {cluster_id}
                Constructs: {cluster_size}
                """
            )

        for r in meta_relation_df.itertuples():

            net.add_edge(
                int(r.source_cluster),
                int(r.target_cluster),
                color=("green" if r.relation == "positive" else "red"),
                value=float(r.support),
                title=f"""
                Dominant sign: {r.relation}
                Positive votes: {r.positive_votes}
                Negative votes: {r.negative_votes}
                Total votes: {r.total_votes}
                Consensus: {r.consensus:.1%}
                """
            )

        # Write net to HTML file and open in browser
        net.write_html(html_file, open_browser=False)
        webbrowser.open("file://" + os.path.abspath(html_file))

    #
    # Static visualization
    #

    else:
        # Construct a networkx graph
        G = nx.DiGraph()

        for cluster_id in visible_clusters:
            G.add_node(
                cluster_id,
                label=(
                    f"{cluster_name_map[cluster_id]}"
                    f"\n(n={cluster_sizes.loc[cluster_id]})"
                )
            )

        for r in meta_relation_df.itertuples():
            G.add_edge(
                int(r.source_cluster),
                int(r.target_cluster),
                color="green" if r.relation == "positive" else "red",
                weight=int(r.support)
            )

        plt.figure(figsize=figsize)

        # Determine node positions
        if len(G.nodes()) > 1:
            pos = nx.kamada_kawai_layout(G, scale=3)
        elif len(G.nodes()) == 1:
            pos = {next(iter(G.nodes())): (0, 0)}
        else:
            pos = {}

        node_colors = [
            "#ffcc00" if node_id in focal_clusters else "#97c2fc"
            for node_id in G.nodes()
        ]

        # Extract edge plotting properties
        edge_colors = [G[u][v]["color"] for u, v in G.edges()]
        edge_widths = [1 + 0.3 * G[u][v]["weight"]for u, v in G.edges()]

        # Draw plot
        nx.draw_networkx_nodes(G, pos, node_size=2200, alpha=0.85, node_color=node_colors)

        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=8)

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.12"
        )

        title = "Global Model"
        if focal_clusters:
            title += (f"\n{len(focal_clusters)} focal clusters")

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

