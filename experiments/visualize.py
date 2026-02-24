import click
import dect
import dect.directions
import matplotlib.pyplot as plt
import torch
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm

from experiments.train_gcn import BaseLineModel
from utils.dataset_loader import SmallGraphDataset
from utils.utility import compute_local_ect
from utils.utility import params2dict
from utils.utility import print_dict
from utils.utility import load_dict_from_json
import os


# Pool node features to a single vector per graph
def pool_node_features(data, pooling_type="mean"):
    if pooling_type == "mean":
        return torch.mean(data, dim=0)  # Average pooling
    elif pooling_type == "sum":
        return torch.sum(data, dim=0)  # Sum pooling
    elif pooling_type == "max":
        return torch.max(data, dim=0).values  # Max pooling
    else:
        raise ValueError("Unsupported pooling type")


@click.command()
@click.option("--dataset-name", default="Letter-high", help="Name of the dataset.")
@click.option("--batch-size", default=32, type=int, help="Batch size for training.")
@click.option(
    "--use-pos-enc", default=False, help="Use positional encoding."
)
@click.option(
    "--ignore-node-features", default=True, help="Ignore node features."
)
@click.option("--use-ect", default=True, help="Use ECT.")
@click.option("--num-thetas", default=16, type=int, help="Number of thetas for ECT.")
@click.option("--ect-hops", default=1, type=int, help="Number of hops for ECT.")
@click.option("--ect-radius", default=1.1, type=float, help="Radius for ECT.")
@click.option("--ect-resolution", default=16, type=int, help="Resolution for ECT.")
@click.option("--ect-scale", default=500, type=int, help="Scale for ECT.")
@click.option(
    "--ect-normalize-before", default=True, help="Normalize ECT before."
)
@click.option(
    "--ect-normalize-after", default=False, help="Normalize ECT after."
)
@click.option("--ect-type", default="edges", help="Type of ECT.")
@click.option("--project-ect", default=False, help="Project ECT.")
@click.option(
    "--projector-path", default="./best_backbone.pt", help="Path to projector model."
)
@click.option("--use-tsne", default="True", help="True to use t-sne | False to use UMAP.")
@click.option(
    "--pooling-type",
    default="sum",
    type=click.Choice(["mean", "sum", "max"], case_sensitive=False),
    help="Pooling type for node features: mean|sum|max.",
)
@click.option(
    "--pos-enc-dim", default=10, type=int, help="Dimension for RWPE."
)
def main(
    dataset_name,
    batch_size,
    use_pos_enc,
    ignore_node_features,
    use_ect,
    num_thetas,
    ect_hops,
    ect_radius,
    ect_resolution,
    ect_scale,
    ect_normalize_before,
    ect_normalize_after,
    ect_type,
    project_ect,
    projector_path,
    use_tsne,
    pooling_type,
    pos_enc_dim,
):
    params_dict = params2dict(
        dataset_name=dataset_name,
        batch_size=batch_size,
        use_pos_enc=use_pos_enc,
        ignore_node_features=ignore_node_features,
        use_ect=use_ect,
        num_thetas=num_thetas,
        ect_hops=ect_hops,
        ect_radius=ect_radius,
        ect_resolution=ect_resolution,
        ect_scale=ect_scale,
        ect_normalize_before=ect_normalize_before,
        ect_normalize_after=ect_normalize_after,
        ect_type=ect_type,
        project_ect=project_ect,
        projector_path=projector_path,
        pooling_type=pooling_type,
        pos_enc_dim = pos_enc_dim,
    )
    print_dict(params_dict)
    
    embed_layer = None

    # Load dataset
    dataset = SmallGraphDataset(
        name=dataset_name,
        batch_size=batch_size,
        seed=42,
        fold=3,
        pos_enc_dim=pos_enc_dim,
    )
    dataset.prepare_data()

    # load projector
    if use_ect:
        v = dect.directions.generate_uniform_directions(
            num_thetas=num_thetas, d=dataset.num_features, seed=0, device="cpu"
        )
        if project_ect:
            print("Loading trained projector for the ECT...")
            model_config = load_dict_from_json(os.path.join(os.path.dirname(projector_path), "params_dict.json"))
            best_model = BaseLineModel(
                input_dim=dataset.num_features,
                num_classes=dataset.num_classes,
                ect_directions=v,
                **model_config,
            )
            # load best validation model
            best_model.load_state_dict(torch.load(projector_path))
            best_model.eval()
            embed_layer = best_model.model.ect_embed_layer
            print(embed_layer)
            # try a forward pass 
            embed_layer(
                torch.rand(
                    num_thetas * ect_resolution,
                )
            )

    train_loader = dataset.train_dataloader()

    # Collect graph representations and labels
    graph_representations = []
    graph_labels = []

    for batch in tqdm(train_loader):
        node_features = []
        if not ignore_node_features:
            node_features.append(batch.x)
        if use_ect:
            ect_features = compute_local_ect(
                x=batch.x,
                edge_index=batch.edge_index,
                v=v,
            )
            if embed_layer is not None:
                with torch.no_grad():
                    ect_features = embed_layer(ect_features).detach()
            node_features.append(ect_features)
        if use_pos_enc:
            node_features.append(batch.pos_enc)
        
        node_features = torch.cat(node_features, axis=1)

        for i in range(batch_size):
            # print(node_features.shape)
            graph_features = node_features[[batch.batch == i]]
            if len(graph_features) > 0:
                graph_features = pool_node_features(graph_features, pooling_type=pooling_type)
                graph_representations.append(graph_features)
                graph_labels.append(batch.y[i])
            # print(batch.y[i])

    graph_representations = torch.stack(graph_representations)
    graph_labels = torch.stack(graph_labels)

    # Use t-SNE instead of UMAP for 2D projection
    visualization_model = (
        TSNE(n_components=2, random_state=42) if use_tsne else umap.UMAP(n_components=2)
    )
    graph_representations_2d = visualization_model.fit_transform(
        graph_representations.cpu().numpy()
    )

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        graph_representations_2d[:, 0],
        graph_representations_2d[:, 1],
        c=graph_labels,
        cmap="viridis",
        s=10,
    )
    plt.colorbar(label="Graph Class")
    plt.title(f"Graph Representations in 2D {'(t-SNE)' if use_tsne else '(UMAP)'}")
    suptitle_str = ""
    if use_ect:
        suptitle_str += "ECT, "
        if project_ect:
            suptitle_str += "learned ECT projector, "
    if use_pos_enc:
        suptitle_str += "Positional Encoding, "
    if ignore_node_features:
        suptitle_str += "Node Features Ignored, "
    suptitle_str += f"Pooling: {pooling_type}"
    plt.suptitle(suptitle_str, fontsize=8, color="gray")  # Subtitle
    plt.xlabel("coord 1")
    plt.ylabel("coord 2")
    plt.show()


if __name__ == "__main__":
    main()
