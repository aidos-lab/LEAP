import os
import random

import click
import dect
import dect.directions
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.model_summary import summarize

from models.baseLineModel import BaseLineModel
from models.modelWrapper import ModelWrapper
from models.modelWrapperRegression import ModelWrapperRegression
from utils.callbacks import SaveBestBackboneCallback, SaveLatestBackboneCallback
from utils.config import Config
from utils.dataset_loader import SmallGraphDataset, get_num_feat_classes
from utils.utility import params2dict, print_dict, save_dict_as_json

LOG_DIR = os.getenv("LOG_DIR", ".")


def training_pipeline(**kwargs):
    force_reload = kwargs.pop("force_reload") if "force_reload" in kwargs else False
    config = Config(**kwargs)
    print(config)

    # Set precision to be the highest possible (can be lowered to accelerate GPU computations)
    torch.set_float32_matmul_precision("highest")

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Build and print parameters dictionary
    params_dict = kwargs

    # For HIV dataset we need to retrieve the number of categories for each feature
    if config.dataset_name == "HIV":
        params_dict["num_embed_feat"] = get_num_feat_classes(config.dataset_name)

    # Set device
    if torch.cuda.is_available():
        accelerator = "cuda"
    elif False and torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    # Load dataset
    dataset = SmallGraphDataset(
        name=config.dataset_name,
        batch_size=config.batch_size,
        seed=config.seed,
        fold=config.fold,
        pos_enc_dim=config.pos_enc_dim,
        pe_name=config.pe_name,
        num_hops=list(config.ect_hops),
    )
    dataset.prepare_data(force_reload=force_reload)

    if config.use_ect:
        if dataset.name == "HIV":
            dim = 3
        else:
            dim = (
                dataset.num_features
                if not config.ect_on_pe
                else config.pos_enc_embed_dim
            )

        ect_directions = dect.directions.generate_uniform_directions(
            num_thetas=config.num_thetas,
            d=dim if not config.ect_on_pe else config.pos_enc_embed_dim,
            seed=config.ect_seed,
            device=accelerator,
        )
    else:
        ect_directions = None

    # Build model
    model = BaseLineModel(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        ect_directions=ect_directions,
        regression=dataset.is_regression,
        **params_dict,
    )

    # Set callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    if dataset.name != "HIV":
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",  # Save the model with the best validation accuracy
            mode="min",
            save_top_k=1,
            filename="best_model",
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_auroc",  # Save the model with the best validation accuracy
            mode="max",
            save_top_k=1,
            filename="best_model",
        )

    save_best_backbone_callback = SaveBestBackboneCallback(
        checkpoint_callback=checkpoint_callback,
    )

    save_latest_backbone_callback = SaveLatestBackboneCallback(
        checkpoint_callback=checkpoint_callback
    )

    if config.wandb_name:
        wandb_logger = pl.loggers.WandbLogger(
            name=config.wandb_name,
            entity="",
            project="",
            log_model=False,
            tags=[
                key for key, val in params_dict.items() if isinstance(val, bool) and val
            ],
        )
        wandb_logger.experiment.config.update(params_dict)

    else:
        wandb_logger = None

    # Set trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=[
            early_stopping,
            lr_monitor,
            checkpoint_callback,
            save_best_backbone_callback,
            save_latest_backbone_callback,
        ],
        logger=wandb_logger,
        accelerator=accelerator,
        default_root_dir=LOG_DIR,
    )

    # Train model
    if dataset.is_regression:
        wrapped_model = ModelWrapperRegression(model, dataset.num_classes)
    else:
        wrapped_model = ModelWrapper(model, dataset.num_classes, dataset.class_ratios)
    summary = summarize(wrapped_model)  # model = your LightningModule instance
    trainable_params = summary.trainable_parameters

    train_result = trainer.fit(
        wrapped_model, dataset.train_dataloader(), dataset.val_dataloader()
    )

    # Save relevant train/val  metrics
    results_dict = dict()
    results_dict["completed_epochs"] = trainer.current_epoch
    results_dict["training_loss"] = trainer.callback_metrics["train_loss"].item()
    results_dict["validation_loss"] = trainer.callback_metrics["val_loss"].item()

    if dataset.is_regression:
        results_dict["training_mse"] = trainer.callback_metrics["train_mse"].item()
        results_dict["validation_mse"] = trainer.callback_metrics["val_mse"].item()
    else:
        results_dict["training_accuracy"] = trainer.callback_metrics[
            "train_accuracy"
        ].item()
        results_dict["validation_accuracy"] = trainer.callback_metrics[
            "val_accuracy"
        ].item()

    # Build best validation model
    best_model = BaseLineModel(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        ect_directions=ect_directions,
        regression=dataset.is_regression,
        **params_dict,
    )

    # load best validation model
    best_model.load_state_dict(
        torch.load(save_best_backbone_callback.best_backbone_path)
    )
    best_model.eval()

    # build and load wrapped model
    if dataset.is_regression:
        best_wrapped_model = ModelWrapperRegression.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            backbone=best_model,
            num_targets=dataset.num_classes,
        )
    else:
        best_wrapped_model = ModelWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            backbone=best_model,
            num_classes=dataset.num_classes,
            class_ratios=dataset.class_ratios,
        )

    best_wrapped_model.eval()

    test_results = trainer.test(best_wrapped_model, dataset.test_dataloader())

    # Save relevant test  metrics
    results_dict["test_loss"] = test_results[0]["test_loss"]
    if dataset.is_regression:
        results_dict["test_mse"] = trainer.callback_metrics["test_mse"].item()
        results_dict["test_r2"] = trainer.callback_metrics["test_r2"].item()
        results_dict["test_mae"] = trainer.callback_metrics["test_mae"].item()
    else:
        results_dict["test_accuracy"] = test_results[0]["test_accuracy"]
        results_dict["test_auroc"] = test_results[0]["test_auroc"]

    # Save parameters to json
    save_dict_as_json(
        params_dict, os.path.join(checkpoint_callback.dirpath, "params_dict.json")
    )

    params_dict["trainable_params"] = trainable_params
    params_dict.update(results_dict)

    if "num_embed_feat" in params_dict:
        del params_dict["num_embed_feat"]

    return params_dict


@click.command()
@click.option("--dataset-name", default=Config.dataset_name, help="Dataset name")
@click.option("--batch-size", default=Config.batch_size, help="Batch size")
@click.option("--hidden-dim", default=Config.hidden_dim, help="Hidden dimension size")
@click.option("--num-layers", default=Config.num_layers, help="Number of layers")
@click.option("--max-epochs", default=Config.max_epochs, help="Number of epochs")
@click.option("--use-ect", default=Config.use_ect, help="Whether to use ECT")
@click.option("--ect-type", default=Config.ect_type, help="ECT type: edges|points")
@click.option(
    "--ect-hops",
    multiple=True,
    type=int,
    default=Config.ect_hops,
    help="List of hops for the neighborhood in l-ECT (e.g., --ect-hops 1 --ect-hops 2)",
)
@click.option(
    "--num-thetas", default=Config.num_thetas, help="Number of directions for the ECT"
)
@click.option("--ect-seed", default=Config.ect_seed, help="Seed for ECT")
@click.option(
    "--use-pos-enc",
    default=Config.use_pos_enc,
    help="Whether to use positional encoding",
)
@click.option(
    "--conv-name",
    default=Config.conv_name,
    help="Convolution layer name: GCNConv|GATConv",
)
@click.option("--pos-enc-dim", default=Config.pos_enc_dim, help="dim of PE")
@click.option(
    "--pos-enc-embed-dim",
    default=Config.pos_enc_embed_dim,
    help="dim of PE fed into the model, if different from pos-enc-dim a linear layer is used to project",
)
@click.option(
    "--exclude-features",
    default=Config.exclude_features,
    help="Set to True to exclude original node features",
)
@click.option(
    "--ect-on-pe",
    default=Config.ect_on_pe,
    help="Set to True to exclude original node features",
)
@click.option(
    "--ect-embed-method",
    default=Config.ect_embed_method,
    help="linear: method to proyect l-ECT",
)
@click.option(
    "--learn-directions",
    default=Config.learn_directions,
    help="Set to True to use deepsets to learn directions of the l-ECT",
)
@click.option(
    "--ect-embed-dim", default=Config.ect_embed_dim, help="Dim to project the l-ECT"
)
@click.option(
    "--ect-scale",
    default=Config.ect_scale,
    help="Scale parameter for the differentiable approx of l-ECT",
)
@click.option(
    "--ect-resolution",
    default=Config.ect_resolution,
    help="Number of discretization steps for computing l-ECT",
)
@click.option("--ect-radius", default=Config.ect_radius, help="Radius for the l-ECT")
@click.option(
    "--ect-normalize-before",
    default=Config.ect_normalize_before,
    help="Normalize local neighbourhoods for l-ECT",
)
@click.option(
    "--ect-normalize-by-graph",
    default=Config.ect_normalize_by_graph,
    help="Normalize by whole graph for l-ECT",
)
@click.option(
    "--model-name", default=Config.model_name, help="GCN | NoMP | GCNwithLECTallSteps"
)
@click.option("--fold", default=Config.fold, help="fold number for cross-validation")
@click.option("--seed", default=Config.seed, help="Random seed for reproducibility")
@click.option(
    "--learnable-pe", default=Config.learnable_pe, help="Keep separate GNN for PEs"
)
@click.option(
    "--wandb-name",
    default=Config.wandb_name,
    help="Name for wandb if empty wandb wont be used",
)
@click.option("--pe-name", default=Config.pe_name, help="rwpe | lape")
@click.option("--force-reload", default=False, help="Recompute pre-transforms")
def main(**kwargs):
    training_pipeline(**kwargs)


if __name__ == "__main__":
    main()
