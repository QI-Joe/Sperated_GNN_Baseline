import argparse
import torch
from utils.data_processing import get_link_prediction_data

# Import the functions for edge candidates and node feature extraction.
from utils.KG_Cluster import generate_new_interactions  # assumed function name


def load_model(checkpoint_path):
    # Load the trained model checkpoint.
    model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.eval()
    # Freeze model parameters.
    for param in model.parameters():
        param.requires_grad = False
    return model


def predict_edges(model, candidate_edges, node_features, threshold=0.5):
    predicted_edges = []
    for edge in candidate_edges:
        u, v = edge
        # Check if both nodes have features.
        if u not in node_features or v not in node_features:
            continue

        # Obtain features for both nodes and prepare the input.
        feat_u = torch.tensor(node_features[u]).float().unsqueeze(0)
        feat_v = torch.tensor(node_features[v]).float().unsqueeze(0)
        # Concatenate features along the feature dimension.
        x = torch.cat([feat_u, feat_v], dim=1)

        # Forward pass through the model.
        output = model(x)
        # Assume output is a logit and apply sigmoid for probability.
        prob = torch.sigmoid(output)

        # If probability exceeds the threshold, mark as prediction 1.
        if prob.item() > threshold:
            predicted_edges.append(edge)
    return predicted_edges


def main():
    parser = argparse.ArgumentParser(
        description="Predict edges using a trained model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file."
    )
    args = parser.parse_args()

    # Load node features using the provided function.
    node_features = get_link_prediction_data()

    # Get candidate edges to be predicted.
    candidate_edges = generate_new_interactions()

    # Load and freeze the model.
    model = load_model(args.checkpoint)

    # Predict connections and gather edges with prediction 1.
    predicted_edges = predict_edges(model, candidate_edges, node_features)

    # Store the predicted edges in a file.
    output_file = "predicted_edges.txt"
    with open(output_file, "w") as f:
        for u, v in predicted_edges:
            f.write(f"{u}\t{v}\n")

    print(f"Prediction complete. {len(predicted_edges)} edges saved to '{output_file}'.")


if __name__ == "__main__":
    main()