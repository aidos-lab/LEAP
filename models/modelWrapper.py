import torch
import pytorch_lightning as pl
import torchmetrics as tm


class ModelWrapper(pl.LightningModule):
    """
    Safe & efficient Lightning wrapper for multiclass classification.

    - Uses NLLLoss (expects backbone to return log-probabilities).
    - Updates metrics per batch; computes them ONCE per epoch.
    - Guards AUROC/AP for degenerate epochs; logs NaN if undefined.
    - Resets metric state & counters every epoch (prevents growth).
    """

    def __init__(self, backbone, num_classes, class_ratios=None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Loss: backbone must output log-probabilities (e.g., via log_softmax)
        self.loss_fn = torch.nn.NLLLoss(weight=class_ratios)

        # Optional auxiliary loss if backbone exposes an aux decoder
        self.lambda_aux = None
        if False and hasattr(backbone, "aux_decoder"):
            self.lambda_aux = 0.1
            self.aux_loss_fn = torch.nn.MSELoss()

        # TorchMetrics (no compute_on_step arg -> compatible with older versions)
        m = dict(task="multiclass", num_classes=num_classes)
        self.tr_acc = tm.Accuracy(**m); self.va_acc = tm.Accuracy(**m); self.te_acc = tm.Accuracy(**m)
        self.tr_ap  = tm.AveragePrecision(**m); self.va_ap  = tm.AveragePrecision(**m); self.te_ap  = tm.AveragePrecision(**m)
        self.tr_auc = tm.AUROC(**m);           self.va_auc = tm.AUROC(**m);           self.te_auc = tm.AUROC(**m)

        # Epoch-level class counters for safe AUROC/AP compute
        self.register_buffer("tr_pos", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("va_pos", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("te_pos", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("tr_tot", torch.zeros(1, dtype=torch.long))
        self.register_buffer("va_tot", torch.zeros(1, dtype=torch.long))
        self.register_buffer("te_tot", torch.zeros(1, dtype=torch.long))

    # ---------- helpers ----------
    @staticmethod
    def _valid_for_roc_ap(pos_counts: torch.Tensor, total: int) -> bool:
        """
        AUROC/AP are undefined if the epoch saw only one class overall.
        We consider it valid if at least one class has both positives and negatives.
        """
        if total <= 0:
            return False
        neg_counts = total - pos_counts
        return ((pos_counts > 0) & (neg_counts > 0)).any().item()

    @staticmethod
    def _to_scalar(x):
        return x.item() if torch.is_tensor(x) and x.numel() == 1 else x

    def _finalize_split(self, prefix, acc, ap, auc, pos_buf, tot_buf):
        # Accuracy is always defined
        acc_val = self._to_scalar(acc.compute())
        self.log(f"{prefix}_accuracy", acc_val, sync_dist=True, prog_bar=(prefix == "train"))

        total = int(tot_buf.item())
        if self._valid_for_roc_ap(pos_buf, total):
            ap_val  = self._to_scalar(ap.compute())
            auc_val = self._to_scalar(auc.compute())
            self.log(f"{prefix}_average_precision", ap_val,  sync_dist=True)
            self.log(f"{prefix}_auroc",             auc_val, sync_dist=True)
        else:
            # Degenerate epoch -> undefined AUROC/AP
            self.log(f"{prefix}_average_precision", float("nan"), sync_dist=True)
            self.log(f"{prefix}_auroc",             float("nan"), sync_dist=True)

        # Reset metric states and counters to prevent growth across epochs
        acc.reset(); ap.reset(); auc.reset()
        pos_buf.zero_(); tot_buf.zero_()

    def _step_common(self, batch, prefix, acc, ap, auc, pos_buf, tot_buf):
        y = batch["y"]  # (B,)

        # Forward + loss
        if self.lambda_aux is None:
            y_log = self.backbone(batch)  # log-probs expected by NLLLoss
            real_loss = self.loss_fn(y_log, y)
            loss = real_loss
        else:
            y_log, aux_x, reconstructed = self.backbone(batch)
            real_loss = self.loss_fn(y_log, y)
            loss = real_loss + self.lambda_aux * self.aux_loss_fn(reconstructed, aux_x)

        # Metrics: update only (no compute here)
        y_prob = y_log.exp()               # convert log-probs -> probs for AUROC/AP
        y_pred = y_log.argmax(dim=-1)
        acc.update(y_pred, y)
        ap.update(y_prob, y)
        auc.update(y_prob, y)

        # O(1) per-batch counters for epoch-level validity checks
        with torch.no_grad():
            counts = torch.bincount(y, minlength=self.num_classes)
            pos_buf += counts
            tot_buf += y.numel()

        bs = y.size(0)
        self.log(f"{prefix}_loss", real_loss, on_step=False, on_epoch=True,
                 batch_size=bs, prog_bar=(prefix == "train"))
        return loss

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train", self.tr_acc, self.tr_ap, self.tr_auc, self.tr_pos, self.tr_tot)

    def validation_step(self, batch, batch_idx):
        return self._step_common(batch, "val",   self.va_acc, self.va_ap, self.va_auc, self.va_pos, self.va_tot)

    def test_step(self, batch, batch_idx):
        return self._step_common(batch, "test",  self.te_acc, self.te_ap, self.te_auc, self.te_pos, self.te_tot)

    def on_train_epoch_end(self):
        self._finalize_split("train", self.tr_acc, self.tr_ap, self.tr_auc, self.tr_pos, self.tr_tot)

    def on_validation_epoch_end(self):
        self._finalize_split("val",   self.va_acc, self.va_ap, self.va_auc, self.va_pos, self.va_tot)

    def on_test_epoch_end(self):
        self._finalize_split("test",  self.te_acc, self.te_ap, self.te_auc, self.te_pos, self.te_tot)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=1e-3)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            ),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
