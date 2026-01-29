import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _canonical_patient_filename(pid) -> str:
    s = str(pid)
    if len(s) >= 2 and s[0].lower() == "p" and s[1:].isdigit():
        return f"p{int(s[1:]):06d}.psv"
    if s.isdigit():
        return f"p{int(s):06d}.psv"
    safe = "".join(ch if ch.isalnum() else "_" for ch in s)
    return f"p{safe}.psv"


def _resolve_path(root_path: str, maybe_rel: str) -> str:
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(root_path, maybe_rel)


def _export_pred_psv_from_big_csv(
    *,
    big_csv_path: str,
    out_dir: str,
    probs_1d: np.ndarray,
    preds_1d: np.ndarray,
    group_col: str = "Patient_ID",
    time_cols: tuple = ("ICULOS", "Hour"),
    sample_step: int = 1,
):
    """
    Writes one PSV per patient with:
      PredictedProbability|PredictedLabel
    Ordering is determined by sorting the big CSV by [Patient_ID, ICULOS, Hour(if present)].
    Alignment is by row order (and optional sample_step).
    """
    df = pd.read_csv(big_csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if group_col not in df.columns:
        raise ValueError(f"CSV missing group_col='{group_col}'. Columns: {list(df.columns)}")

    sort_cols = [group_col] + [c for c in time_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if sample_step > 1:
        df = df.iloc[::sample_step].reset_index(drop=True)

    probs_1d = np.asarray(probs_1d).reshape(-1)
    preds_1d = np.asarray(preds_1d).reshape(-1)

    if len(df) != len(probs_1d) or len(df) != len(preds_1d):
        raise ValueError(
            f"PSV export length mismatch:\n"
            f"  rows in CSV(after sample_step) = {len(df)}\n"
            f"  probs length                   = {len(probs_1d)}\n"
            f"  preds length                   = {len(preds_1d)}\n"
            f"Check sample_step and that inference used shuffle=False."
        )

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # We only need group_col to split; time ordering is already baked into df sorting.
    pids = df[group_col].to_numpy()

    # Write per patient efficiently
    # (manual writing avoids pandas overhead and controls formatting)
    start = 0
    n = len(pids)

    while start < n:
        pid = pids[start]
        end = start + 1
        while end < n and pids[end] == pid:
            end += 1

        fname = _canonical_patient_filename(pid)
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write("PredictedProbability|PredictedLabel\n")
            for pr, lb in zip(probs_1d[start:end], preds_1d[start:end]):
                f.write(f"{float(pr):.6f}|{int(lb)}\n")

        start = end


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self, sample_logits=None):
        # BCE if logits (B,1)
        if sample_logits is not None and hasattr(sample_logits, "shape"):
            if sample_logits.dim() >= 2 and sample_logits.shape[-1] == 1:
                pw = float(getattr(self.args, "pos_weight", 0.0))
                if pw > 0:
                    pos_w = torch.tensor([pw], device=self.device, dtype=torch.float32)
                    print(f"[LOSS] BCEWithLogitsLoss(pos_weight={pw:.4f})")
                    return nn.BCEWithLogitsLoss(pos_weight=pos_w)
                print("[LOSS] BCEWithLogitsLoss (unweighted)")
                return nn.BCEWithLogitsLoss()

        print("[LOSS] CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    def _build_model(self):
        train_data, train_loader = self._get_data(flag="TRAIN")
        self.args.pred_len = 0  # classification

        # infer enc_in
        if hasattr(train_data, "feature_df"):
            self.args.enc_in = int(train_data.feature_df.shape[1])
        elif hasattr(train_data, "feature_cols"):
            self.args.enc_in = int(len(train_data.feature_cols))
        else:
            bx = next(iter(train_loader))[0]
            self.args.enc_in = int(bx.shape[-1])

        # infer num_class (mostly irrelevant for BCE path)
        if hasattr(train_data, "class_names"):
            self.args.num_class = int(len(train_data.class_names))
        else:
            self.args.num_class = int(getattr(self.args, "num_class", 2))

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _forward_batch(self, batch_x, padding_mask):
        batch_x = batch_x.float().to(self.device)
        padding_mask = padding_mask.float().to(self.device)
        return self.model(batch_x, padding_mask, None, None)

    def _compute_loss_and_preds(self, logits, labels, criterion, threshold=0.5):
        y = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        # BCE
        if logits.shape[-1] == 1:
            y_float = y.float().view(-1, 1).to(self.device)
            loss = criterion(logits, y_float)

            prob_pos = torch.sigmoid(logits).view(-1)
            pred = (prob_pos >= float(threshold)).long()
            true = y.view(-1).long().to(self.device)
            conf = torch.where(pred == 1, prob_pos, 1.0 - prob_pos)

            return (
                loss,
                pred.detach().cpu().numpy(),
                true.detach().cpu().numpy(),
                conf.detach().cpu().numpy(),
                prob_pos.detach().cpu().numpy(),
            )

        # CE
        y_long = y.view(-1).long().to(self.device)
        loss = criterion(logits, y_long)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        conf = probs.max(dim=-1).values

        prob_pos = probs[:, 1].detach().cpu().numpy() if probs.shape[-1] == 2 else None

        return (
            loss,
            pred.detach().cpu().numpy(),
            y_long.detach().cpu().numpy(),
            conf.detach().cpu().numpy(),
            prob_pos,
        )

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        all_pred, all_true = [], []

        with torch.no_grad():
            for batch in vali_loader:
                if len(batch) == 3:
                    batch_x, label, padding_mask = batch
                else:
                    batch_x, label = batch[0], batch[1]
                    padding_mask = batch[2] if len(batch) > 2 else torch.ones(batch_x.shape[0], batch_x.shape[1])

                logits = self._forward_batch(batch_x, padding_mask)
                loss, pred_np, true_np, _, _ = self._compute_loss_and_preds(logits, label, criterion)

                losses.append(loss.item())
                all_pred.append(pred_np)
                all_true.append(true_np)

        loss_mean = float(np.mean(losses)) if losses else np.nan
        pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
        true_all = np.concatenate(all_true, axis=0) if all_true else np.array([])
        acc = cal_accuracy(pred_all, true_all) if len(pred_all) else 0.0

        self.model.train()
        return loss_mean, acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")

        ckpt_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_path, exist_ok=True)

        results_dir = os.path.join("./results", setting)
        os.makedirs(results_dir, exist_ok=True)

        curve_dir = os.path.join(results_dir, "curves")
        os.makedirs(curve_dir, exist_ok=True)

        model_optim = self._select_optimizer()

        # choose criterion by probing one forward pass
        first_batch = next(iter(train_loader))
        if len(first_batch) == 3:
            bx0, y0, pm0 = first_batch
        else:
            bx0, y0 = first_batch[0], first_batch[1]
            pm0 = first_batch[2] if len(first_batch) > 2 else torch.ones(bx0.shape[0], bx0.shape[1])

        with torch.no_grad():
            logits0 = self._forward_batch(bx0, pm0)
        criterion = self._select_criterion(logits0)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        train_steps = len(train_loader)

        hist = {"epoch": [], "train_loss": [], "val_loss": []}
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            train_losses = []

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                if len(batch) == 3:
                    batch_x, label, padding_mask = batch
                else:
                    batch_x, label = batch[0], batch[1]
                    padding_mask = batch[2] if len(batch) > 2 else torch.ones(batch_x.shape[0], batch_x.shape[1])

                logits = self._forward_batch(batch_x, padding_mask)
                loss, _, _, _, _ = self._compute_loss_and_preds(logits, label, criterion)

                loss.backward()
                cg = float(getattr(self.args, "clip_grad", 0.0))
                if cg > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cg)
                model_optim.step()
                train_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - (i + 1))
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.6f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.1f}s")
                    iter_count = 0
                    time_now = time.time()

            train_loss = float(np.mean(train_losses)) if train_losses else np.nan
            val_loss, val_acc = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.1f}s")
            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            early_stopping(val_loss, self.model, ckpt_path)

            hist["epoch"].append(epoch + 1)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            pd.DataFrame(hist).to_csv(os.path.join(curve_dir, "learning_curves.csv"), index=False)

            plt.figure()
            plt.plot(hist["epoch"], hist["train_loss"], label="train")
            plt.plot(hist["epoch"], hist["val_loss"], label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(curve_dir, "loss_curve.png"), dpi=150)
            plt.close()

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # load best checkpoint
        best_model_path = os.path.join(ckpt_path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # After training: always run inference on TEST and THRESH
        self.test(setting, test=1, flag="TEST", test_tag="TEST")
        self.test(setting, test=1, flag="THRESH", test_tag="TRAIN_THRESH")

        return self.model

    def test(self, setting, test=0, flag="TEST", test_tag=None):
        """
        test=1 -> load best checkpoint from ./checkpoints/<setting>/checkpoint.pth
        flag   -> TEST / THRESH / VAL / TRAIN (only TEST+THRESH typically used)
        test_tag -> output suffix; defaults to flag.upper()
        """
        tag = (test_tag if test_tag is not None else str(flag)).upper()

        # loader
        test_data, test_loader = self._get_data(flag=flag)

        # load ckpt if requested
        if test:
            print("loading model")
            ckpt = os.path.join("./checkpoints", setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        results_dir = os.path.join("./results", setting)
        os.makedirs(results_dir, exist_ok=True)

        self.model.eval()

        all_logits = []
        all_true = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    batch_x, label, padding_mask = batch
                else:
                    batch_x, label = batch[0], batch[1]
                    padding_mask = batch[2] if len(batch) > 2 else torch.ones(batch_x.shape[0], batch_x.shape[1])

                logits = self._forward_batch(batch_x, padding_mask)
                all_logits.append(logits.detach().cpu())
                all_true.append(label.detach().cpu() if isinstance(label, torch.Tensor) else torch.tensor(label))

        logits_t = torch.cat(all_logits, dim=0)
        trues_t = torch.cat(all_true, dim=0)

        print(f"[{tag}] shape:", logits_t.shape, trues_t.shape)

        # Convert & compute
        if logits_t.dim() >= 2 and logits_t.shape[-1] == 1:
            logits_np = logits_t.numpy().reshape(-1)
            probs_1d = _sigmoid(logits_np)
            preds_1d = (probs_1d >= 0.5).astype(np.int64)
            true_np = trues_t.view(-1).long().numpy()
            conf_1d = np.where(preds_1d == 1, probs_1d, 1.0 - probs_1d)

            probs_save = probs_1d.reshape(-1, 1)
            logits_save = logits_np.reshape(-1, 1)
        else:
            logits_np = logits_t.numpy()
            probs = F.softmax(logits_t, dim=-1).numpy()
            preds_1d = np.argmax(probs, axis=-1).astype(np.int64)
            true_np = trues_t.view(-1).long().numpy()
            conf_1d = np.max(probs, axis=-1)
            probs_1d = probs[:, 1] if probs.shape[-1] == 2 else None

            probs_save = probs
            logits_save = logits_np

        # Save tagged outputs (no overwrites)
        np.save(os.path.join(results_dir, f"logits_{tag}.npy"), logits_save)
        np.save(os.path.join(results_dir, f"probs_{tag}.npy"), probs_save)
        np.save(os.path.join(results_dir, f"pred_{tag}.npy"), preds_1d)
        np.save(os.path.join(results_dir, f"true_{tag}.npy"), true_np)

        out_df = {
            "sample_idx": np.arange(len(preds_1d), dtype=np.int64),
            "true": true_np,
            "pred": preds_1d,
            "conf": conf_1d,
        }
        if probs_1d is not None:
            out_df["prob_pos"] = probs_1d
        pd.DataFrame(out_df).to_csv(os.path.join(results_dir, f"predictions_{tag}.csv"), index=False)

        acc = cal_accuracy(preds_1d, true_np)
        print(f"[{tag}] accuracy: {acc}")

        with open(os.path.join(results_dir, "result_classification.txt"), "a", encoding="utf-8") as f:
            f.write(setting + "\n")
            f.write(f"{tag} accuracy:{acc}\n\n")

        # ---- NEW: write PSV predictions automatically (CSV->PSV) ----
        # Only do this when we have a prob_pos (binary) and when flag is TEST/THRESH.
        if probs_1d is not None and tag in ("TEST", "TRAIN_THRESH", "THRESH"):
            group_col = str(getattr(self.args, "group_col", "Patient_ID"))
            time_cols = ("ICULOS", "Hour")
            sample_step = int(getattr(self.args, "sample_step", 1))

            # choose which CSV to use for mapping
            if tag == "TEST":
                csv_rel = getattr(self.args, "test_dir", None)
            else:
                csv_rel = getattr(self.args, "thresh_dir", None)

            if csv_rel is None:
                print(f"[WARN] No csv path configured for tag={tag}, skipping PSV export.")
                return

            csv_path = _resolve_path(self.args.root_path, csv_rel)
            out_psv_dir = os.path.join(results_dir, f"pred_psv_{tag}_WORK")

            try:
                _export_pred_psv_from_big_csv(
                    big_csv_path=csv_path,
                    out_dir=out_psv_dir,
                    probs_1d=probs_1d,
                    preds_1d=preds_1d,
                    group_col=group_col,
                    time_cols=time_cols,
                    sample_step=sample_step,
                )
                print(f"[{tag}] wrote per-patient PSV -> {out_psv_dir}")
            except Exception as e:
                print(f"[WARN] PSV export failed for {tag}: {e}")

        return
