import torch
import pytorch_lightning as pl
import torchmetrics as tm


class ModelWrapperRegression(pl.LightningModule):
    """Wrapper class for learning on graphs.

    The purpose of this wrapper is to permit learning representations
    with various internal models, which we refer to as backbones. The
    wrapper provides a consistent training procedure.
    """

    def __init__(self, backbone, num_targets):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = torch.nn.MSELoss()


        self.train_mse = tm.MeanSquaredError()
        self.validation_mse = tm.MeanSquaredError()
        self.test_mse = tm.MeanSquaredError()

        self.train_mae = tm.MeanAbsoluteError()
        self.validation_mae = tm.MeanAbsoluteError()
        self.test_mae = tm.MeanAbsoluteError()

        self.train_r2 = tm.R2Score()
        self.validation_r2 = tm.R2Score()
        self.test_r2 = tm.R2Score()

    def step(self, batch, batch_idx, prefix, mse, mae, r2):
        y = batch["y"]

        y_pred = self.backbone(batch)
        loss = self.loss_fn(y_pred, y)

        mse.update(y_pred, y)
        mae.update(y_pred, y)
        r2.update(y_pred, y)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
            prog_bar=prefix == "train",
        )

        self.log(
            f"{prefix}_mse",
            mse,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        self.log(
            f"{prefix}_mae",
            mae,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
        )

        self.log(
            f"{prefix}_r2",
            r2,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch),
            prog_bar=prefix == "train",
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "train",
            self.train_mse,
            self.train_mae,
            self.train_r2,
        )

    def validation_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "val",
            self.validation_mse,
            self.validation_mae,
            self.validation_r2,
        )

    def test_step(self, batch, batch_idx):
        return self.step(
            batch,
            batch_idx,
            "test",
            self.test_mse,
            self.test_mae,
            self.test_r2,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=1e-3)

        # Add a scheduler that halves the learning rate as soon as the
        # validation loss starts plateauing. This is available to each
        # model so it does not put as at an unfair advantage.
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            ),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }

        return [optimizer], [scheduler]