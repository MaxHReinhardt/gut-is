import pandas as pd


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

