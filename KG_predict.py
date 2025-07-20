import argparse
import torch
import pandas as pd
import os
import torch.nn as nn
import numpy as np
import sys

# Assuming these are the correct import paths based on your project structure
from utils.data_processing import get_link_prediction_data4KG_Data
from utils.KG_Cluster import (
    get_cluster_centre_points,
    evaluate_clusters,
    classify_clusters,
    load_graph_from_csv,
    generate_new_interactions,
)
from models.GAT_Standard import GAT_Simple
from models.GraphSage_Standard import GraphSage_Simple
from models.GCONV import GCONV_Simple
from models.modules import MergeLayer
from utils.utils import get_parameter_sizes

# Import the functions for edge candidates and node feature extraction.
from utils.KG_Cluster import generate_new_interactions  # assumed function name


def load_model_and_checkpoint(model_name: str, in_channels: int, out_channels: int, checkpoint_path: str, device: str, args):
    """
    Loads the model architecture and the trained weights from a checkpoint.
    """
    # Define model architecture based on name
    shared_kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'device': device, 'dropout': args.dropout}
    match model_name:
        case "GCN":
            dynamic_backbone = GCONV_Simple(
                hidden_dim1=args.hidden_channels, hidden_dim2=args.hidden_channels, time_dim=1, **shared_kwargs)
        case "GAT":
            dynamic_backbone = GAT_Simple(
                num_layers=args.num_layers, heads=args.num_heads, negative_slope=args.negative_slope,
                add_self_loops=True, time_dim=1, **shared_kwargs)
        case "GraphSAGE":
            dynamic_backbone = GraphSage_Simple(
                hidden_channels=args.hidden_channels, time_dim=1, **shared_kwargs)
        case _:
            raise NotImplementedError(f"Model {model_name} is not implemented.")

    link_predictor = MergeLayer(input_dim1=out_channels, input_dim2=out_channels, hidden_dim=out_channels, output_dim=1).to(device)
    model = nn.Sequential(dynamic_backbone, link_predictor).to(device)

    print(f'model -> {model}')
    print(f'model name: {model_name}, #parameters: {get_parameter_sizes(model) * 4} B')

    # Load the saved state dictionary
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def predict_links(model: nn.Module, candidate_edges: torch.Tensor, node_features: torch.Tensor, edge_index: torch.Tensor, device: str, threshold=0.5):
    """
    Predicts links for a set of candidate edges using a GNN model.
    """
    predicted_edges = []
    with torch.no_grad():
        # Get node embeddings from the GNN backbone
        node_embeddings = model[0].forward_predict(node_features.to(device), edge_index.to(device))

        # Iterate through candidate edges to predict
        for u, v in candidate_edges.T:
            # Get embeddings for the source and destination nodes
            emb_u = node_embeddings[u].unsqueeze(0)
            emb_v = node_embeddings[v].unsqueeze(0)

            # Pass embeddings to the link predictor head
            output = model[1](emb_u, emb_v)
            prob = torch.sigmoid(output).item()

            if prob > threshold:
                predicted_edges.append((u.item(), v.item(), prob))

    print(f"generated {len(predicted_edges)} predicted edges over {candidate_edges.shape[1]} with probability > {threshold}.")
    return predicted_edges


def main():
    parser = argparse.ArgumentParser(
        description="Predict edges using a trained model checkpoint."
    )
    parser.add_argument("--models", type=str, default="GCN", help="Model for training.")
    parser.add_argument("--dataset", type=str, default="entities", help="Dataset for training.")
    parser.add_argument("--dataset_start", type=int, default=0, help="Start index for dataset.")
    parser.add_argument("--dataset_end", type=int, default=400, help="End index for dataset.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels.")
    
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--prediction_threshold", type=float, default=0.51, help="Probability threshold to classify a link as positive.")
    args = parser.parse_args()

    dataset_path = {
        "general": r"../Standard_Dataset/kg/",
        "node": r"extraction_results_{}_{}_nodes.csv".format(args.dataset_start, args.dataset_end),
        "edge": r"extraction_results_{}_{}_edges.csv".format(args.dataset_start, args.dataset_end),
        "node_feat": r"embedding_result/{}_{}_embeddings.pt".format(args.dataset_start, args.dataset_end),
        "edge_feat": r"embedding_result/{}_{}_edge_embeddings.pt".format(args.dataset_start, args.dataset_end)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Cluster nodes and find important ones
    G = load_graph_from_csv(csv_path=os.path.join(dataset_path["general"], dataset_path["edge"]))
    candidate_edges = generate_new_interactions(G, top_k_large=3)
    candidate_edge_tensor = np.array(list(zip(*candidate_edges)), dtype=np.int64)

    candidate_edge_tensor = torch.from_numpy(candidate_edge_tensor).to(device)
    print(f"Generated {candidate_edge_tensor.shape[1]} candidate edges for prediction.")

    # Load graph data for the model
    node_raw_features, _, data_list = get_link_prediction_data4KG_Data(dataset_name=args.dataset, snapshot=3, kg_path=dataset_path)
    full_data, _, _, _, _, _ = data_list
    edges = np.vstack([full_data.src_node_ids, full_data.dst_node_ids]).astype(np.int64)
    full_graph_edge_index = torch.from_numpy(edges)
    
    # --- 2. Load Model ---
    checkpoint_path = fr"saved_models/{args.models}/{args.dataset}/{args.models}_seed2025/{args.models}_seed2025.pkl"
    model = load_model_and_checkpoint(
        model_name=args.models,
        in_channels=node_raw_features.shape[1],
        out_channels=args.hidden_channels,
        checkpoint_path=checkpoint_path,
        device=device,
        args=args
    )

    # --- 3. Predict Links ---
    predicted_edges = predict_links(
        model=model,
        candidate_edges=candidate_edge_tensor,
        node_features=torch.from_numpy(node_raw_features).float(),
        edge_index=full_graph_edge_index.long(),
        device=device,
        threshold=args.prediction_threshold
    )

    # --- 4. Save Results ---
    if len(predicted_edges) == 0:
        print("No edges predicted above the threshold.")
        sys.exit(0)
    predicted_edges = np.array(predicted_edges)
    predicted_edges = predicted_edges[predicted_edges[:, 2].argsort()[::-1]]
    if not os.path.exists("KG_predictions"):
        os.makedirs("KG_predictions")
    output_file = f"KG_predictions/predicted_edges_{args.models}_{args.dataset_start}_{args.dataset_end}.csv"
    pd_edges = pd.DataFrame(predicted_edges[:100,], columns=["src", "dst", "probability"])
    pd_edges.to_csv(output_file, index=False)

    print(f"Prediction complete. {len(predicted_edges)} new links saved to '{output_file}'.")


if __name__ == "__main__":
    main()