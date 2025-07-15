import pandas as pd
import networkx as nx
import argparse
import os

"""
KG_Cluster.py

This script shows how to load a graph stored in CSV files, classify clusters (sub-graphs)
using traditional graph algorithms (connected components), and then locate multiple center points for each cluster.
It demonstrates:
1. Loading graph edges from a CSV file.
2. Classifying the graph into clusters/sub-graphs using connected component detection.
3. Selecting top-k center points for each cluster using a centrality measure.
4. Detailed explanation is provided in the comments.

Assumptions:
- The CSV file has at least two columns: 'source' and 'target' (representing an edge from source to target).
- Optionally, additional columns such as 'weight' can be provided; if so, the weight will be used.
- The "center points" are selected based on a centrality measure (in this example: degree centrality).
- Clusters are considered as connected components in an undirected graph.
"""

SMALL_CLUSTER_THRESHOLD = 10
LARGE_CLUSTER_THRESHOLD = 20

def load_graph_from_csv(csv_path: str) -> nx.Graph:
    """
    Load the graph from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing the graph edges.

    Returns:
        G: A NetworkX graph built from the CSV data.
        
    The CSV file is expected to have columns: 'source', 'target', and optionally 'weight'.
    If weight is provided, it will be added as an edge attribute.
    """
    df = pd.read_csv(csv_path)
    
    # Check if weight column exists
    has_weight = 'weight' in df.columns
    
    # Create an undirected graph
    G = nx.Graph()
    if has_weight:
        for _, row in df.iterrows():
            G.add_edge(row['Source'], row['Target'], weight=row['weight'])
    else:
        for _, row in df.iterrows():
            G.add_edge(row['Source'], row['Target'])
    
    return G

def classify_clusters(G: nx.Graph) -> list:
    """
    Classify the graph into clusters (i.e., connected components).
    
    Args:
        G: A NetworkX graph.
        
    Returns:
        A list of subgraphs, each representing a cluster.
    """
    # Each connected component is returned as a set of nodes
    clusters = list(nx.connected_components(G))
    
    # Convert each cluster of nodes into a subgraph for further analysis.
    cluster_subgraphs = [G.subgraph(nodes).copy() for nodes in clusters]
    return cluster_subgraphs

def get_cluster_centre_points(cluster: nx.Graph, method: str = 'degree', top_k: int = 3) -> list:
    """
    Locates the centre points in a cluster using various centrality measures.
    
    This function computes a centrality measure for the nodes in a cluster.
    Available methods:
      - 'degree': degree centrality.
      - 'closeness': closeness centrality.
      - 'betweenness': betweenness centrality.
      - 'eigenvector': eigenvector centrality.
      - 'long_connect': Identify important nodes that act as bridges within the graph.
    
        - In this scenario, these nodes appear as articulation points (i.e., their removal increases
        the number of connected components), often connecting two large clusters or linking a small,
        dense cluster to a larger cluster.
    
    The function then selects the top-k nodes with the highest centrality scores.
    
    Args:
        cluster: A NetworkX subgraph (cluster).
        method: The centrality method to use.
        top_k: Number of centre points to extract.
        
    Returns:
        A list of nodes representing the centre points of the cluster.
    """
    number_of_nodes = cluster.number_of_nodes()
    if number_of_nodes > LARGE_CLUSTER_THRESHOLD:
        method = "long_connect"
    
    if method == 'degree':
        centrality = nx.degree_centrality(cluster)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(cluster)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(cluster)
    elif method == "long_connect":
        centrality = list(nx.articulation_points(cluster))
    elif method == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(cluster)
        except nx.NetworkXException:
            # Fallback to degree centrality if eigenvector centrality fails
            centrality = nx.degree_centrality(cluster)
    else:
        # Default to degree centrality if unknown method is provided
        centrality = nx.degree_centrality(cluster)
    
    if isinstance(centrality, dict):
        sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    else:
        sorted_nodes = sorted(enumerate(centrality), key=lambda item: item[1], reverse=True)
    top_centres = [node for node, _ in sorted_nodes[:top_k]]
    return top_centres

def generate_new_interactions(G: nx.Graph, top_k_large: int = 3) -> list:
    """
    Generate new interactions (edges) between clusters based on their size.
    
    For clusters with less than 10 nodes ("small"), only the first centre point is used.
    For clusters with more than 20 nodes ("large"), the top-k centre points are used.
    
    New interactions are generated as follows:
        - Between two small clusters: an edge is created using their single (first) centre points.
        - Between a small and a large cluster or between two large clusters:
        edges are created for every combination of the available centre points.
        
    Args:
        G: A NetworkX graph.
        top_k_large: Number of centre points to consider for large clusters.
    
    Returns:
        A list of tuples representing new interactions (edges) between centre points.
    """
    clusters = classify_clusters(G)
    centres_by_cluster = []
    
    # Identify clusters and pick centre points accordingly.
    for cluster in clusters:
        num_nodes = cluster.number_of_nodes()
        if num_nodes < SMALL_CLUSTER_THRESHOLD:
            centres = get_cluster_centre_points(cluster, method='degree', top_k=1)
            centres_by_cluster.append(('small', centres))
        elif num_nodes > LARGE_CLUSTER_THRESHOLD:
            centres = get_cluster_centre_points(cluster, method='degree', top_k=top_k_large)
            centres_by_cluster.append(('large', centres))
    
    new_interactions = set()
    
    # Generate new interactions between eligible clusters.
    for i in range(len(centres_by_cluster)):
        type_i, centres_i = centres_by_cluster[i]
        for j in range(i + 1, len(centres_by_cluster)):
            type_j, centres_j = centres_by_cluster[j]
            if type_i == 'small' and type_j == 'small':
                # Only the first centre points are used.
                new_interactions.add((centres_i[0], centres_j[0]))
            else:
                # For small-large and large-large, create interactions for every combination.
                for node_i in centres_i:
                    for node_j in centres_j:
                        new_interactions.add((node_i, node_j))
    
    return list(new_interactions)


def evaluate_clusters(G: nx.Graph, top_k: int = 3) -> dict:
    """
    Evaluate clusters by classifying, then finding centre points (top-k for each cluster).
    
    Args:
        G: A NetworkX graph.
        top_k: Number of centre points to extract for each cluster.
        
    Returns:
        cluster_evaluation: A dictionary where keys are cluster IDs (int) and values
                            are dictionaries with information about the cluster:
                            - size: number of nodes
                            - centre_points: list of top-k centre point nodes.
    """
    clusters = classify_clusters(G)
    cluster_evaluation = {}
    
    for idx, cluster in enumerate(clusters):
        centres = get_cluster_centre_points(cluster, top_k=top_k)
        cluster_evaluation[idx] = {
            'size': cluster.number_of_nodes(),
            'centre_points': centres
        }
    return cluster_evaluation



def main():
    
    parser = argparse.ArgumentParser(description="KG_Cluster: Cluster evaluation of graph entities.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of centre points to extract for each cluster.")
    args = parser.parse_args()
    
    path = r"../Standard_Dataset/KG/"

    # Automatically trace files in the folder that end with "edges" or contain "edges" in their name
    edges_files = ""
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) and 'edges' in filename:
            edges_files= full_path
    
    # Load graph from CSV file
    G = load_graph_from_csv(edges_files)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Evaluate clusters (classification and extraction of centre points)
    cluster_info = evaluate_clusters(G, args.top_k)
    new_interaction = generate_new_interactions(G, top_k_large=args.top_k)
    
    # Print results
    for cluster_id, info in cluster_info.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {info['size']} nodes")
        print(f"  Top-{args.top_k} centre point(s): {info['centre_points']}")
        
    print(f"\nNew interactions generated between clusters: {len(new_interaction)} edges.")

if __name__ == '__main__':
    main()