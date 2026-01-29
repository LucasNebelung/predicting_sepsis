# exp/exp_classification.py
# FULL REPLACEMENT
# - Supports --pos_weight for BCEWithLogitsLoss (binary logits shape (B,1))
# - Writes ONLY ONE plot: train loss + val loss
#   -> ./results/<setting>/curves/loss_curve.png
#   -> ./results/<setting>/curves/learning_curves.csv

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


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    # --------------------------
    # helpers
    # --------------------------
    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self, sample_logits=None, sample_labels=None):
        """
        - If logits are (B, 1): BCEWithLogitsLoss (optionally with pos_weight)
        - If logits are (B, C>=2): CrossEntropyLoss
        """
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

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')

        # infer input sizes from dataset
        self.args.seq_len = getattr(self.args, "seq_len", 48)
        self.args.pred_len = 0

        # feature dimension
        if hasattr(train_data, "feature_df"):
            self.args.enc_in = train_data.feature_df.shape[1]
        elif hasattr(train_data, "feature_cols"):
            self.args.enc_in = len(train_data.feature_cols)
        else:
            bx = next(iter(train_loader))[0]
            self.args.enc_in = bx.shape[-1]

        # number of classes
        if hasattr(train_data, "class_names"):
            self.args.num_class = len(train_data.class_names)
        elif hasattr(train_data, "num_class"):
            self.args.num_class = int(train_data.num_class)
        else:
            self.args.num_class = int(getattr(self.args, "num_class", 2))

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # --------------------------
    # forward / loss
    # --------------------------
    def _forward_batch(self, batch_x, padding_mask):
        batch_x = batch_x.float().to(self.device)
        padding_mask = padding_mask.float().to(self.device)
        outputs = self.model(batch_x, padding_mask, None, None)
        return outputs

    def _compute_loss_and_preds(self, logits, labels, criterion, threshold=0.5):
        """
        Supports:
          - CE: logits (B,C), labels (B,) or (B,1)
          - BCE: logits (B,1), labels (B,) or (B,1) or float

        Returns: loss, pred_np, true_np, conf_np, prob_pos_np (binary when possible)
        """
        y = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        # BCE path (binary with 1 logit)
        if logits.shape[-1] == 1:
            y_float = y.float().view(-1, 1).to(self.device)
            loss = criterion(logits, y_float)

            prob_pos = torch.sigmoid(logits).view(-1)  # (B,)
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

        # CE path (multi-class or 2-class with 2 logits)
        y_long = y.view(-1).long().to(self.device)
        loss = criterion(logits, y_long)

        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        conf = probs.max(dim=-1).values

        prob_pos = None
        if logits.shape[-1] == 2:
            prob_pos = probs[:, 1]

        return (
            loss,
            pred.detach().cpu().numpy(),
            y_long.detach().cpu().numpy(),
            conf.detach().cpu().numpy(),
            None if prob_pos is None else prob_pos.detach().cpu().numpy(),
        )

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_losses = []
        all_pred = []
        all_true = []

        with torch.no_grad():
            for batch in vali_loader:
                if len(batch) == 3:
                    batch_x, label, padding_mask = batch
                else:
                    batch_x, label = batch[0], batch[1]
                    padding_mask = batch[2] if len(batch) > 2 else torch.ones(batch_x.shape[0], batch_x.shape[1])

                logits = self._forward_batch(batch_x, padding_mask)
                loss, pred_np, true_np, _, _ = self._compute_loss_and_preds(logits, label, criterion)

                total_losses.append(loss.item())
                all_pred.append(pred_np)
                all_true.append(true_np)

        total_loss = float(np.mean(total_losses)) if total_losses else np.nan
        pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
        true_all = np.concatenate(all_true, axis=0) if all_true else np.array([])

        acc = cal_accuracy(pred_all, true_all) if len(pred_all) else 0.0
        self.model.train()
        return total_loss, acc

    # --------------------------
    # training
    # --------------------------
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')

        ckpt_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_path, exist_ok=True)

        results_dir = os.path.join('./results', setting)
        os.makedirs(results_dir, exist_ok=True)

        curve_dir = os.path.join(results_dir, 'curves')
        os.makedirs(curve_dir, exist_ok=True)

        model_optim = self._select_optimizer()

        # pick criterion from a single batch (so we don’t guess wrong)
        first_batch = next(iter(train_loader))
        if len(first_batch) == 3:
            bx0, y0, pm0 = first_batch
        else:
            bx0, y0 = first_batch[0], first_batch[1]
            pm0 = first_batch[2] if len(first_batch) > 2 else torch.ones(first_batch[0].shape[0], first_batch[0].shape[1])

        with torch.no_grad():
            logits0 = self._forward_batch(bx0, pm0)
        criterion = self._select_criterion(logits0, y0)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        time_now = time.time()
        train_steps = len(train_loader)

        # ONLY these for plotting
        hist = {"epoch": [], "train_loss": [], "val_loss": []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []
            train_pred = []
            train_true = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                if len(batch) == 3:
                    batch_x, label, padding_mask = batch
                else:
                    batch_x, label = batch[0], batch[1]
                    padding_mask = batch[2] if len(batch) > 2 else torch.ones(batch_x.shape[0], batch_x.shape[1])

                logits = self._forward_batch(batch_x, padding_mask)

                loss, pred_np, true_np, _, _ = self._compute_loss_and_preds(logits, label, criterion)

                loss.backward()
                
                cg = float(getattr(self.args, "clip_grad", 0.0))
                if cg > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cg)
                model_optim.step()

                train_losses.append(loss.item())
                train_pred.append(pred_np)
                train_true.append(true_np)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.6f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.1f}s")
                    iter_count = 0
                    time_now = time.time()

            train_loss = float(np.mean(train_losses)) if train_losses else np.nan
            train_pred_all = np.concatenate(train_pred, axis=0) if train_pred else np.array([])
            train_true_all = np.concatenate(train_true, axis=0) if train_true else np.array([])
            train_acc = cal_accuracy(train_pred_all, train_true_all) if len(train_pred_all) else 0.0

            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.1f}")
            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} | "
                f"Vali Loss: {vali_loss:.3f} Vali Acc: {vali_acc:.3f} | "
                f"Test Loss: {test_loss:.3f} Test Acc: {test_acc:.3f}"
            )

            # early stopping
            early_stopping(vali_loss, self.model, ckpt_path)
            if early_stopping.early_stop:
                print("Early stopping")

            # --- record ONLY train + val loss ---
            hist["epoch"].append(epoch + 1)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(vali_loss)

            # --- save CSV ---
            pd.DataFrame(hist).to_csv(os.path.join(curve_dir, "learning_curves.csv"), index=False)

            # --- save ONLY ONE plot: train vs val loss ---
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
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # load best checkpoint
        best_model_path = os.path.join(ckpt_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # After training: always produce TEST preds and THRESH preds
        self.test(setting, test=1, flag="TEST", test_tag="TEST")
        self.test(setting, test=1, flag="THRESH", test_tag="TRAIN_THRESH")

    # --------------------------
    # test / inference
    # --------------------------
    def _write_sepsis_pred_psv_dir(self, dataset, prob_pos, out_dir, default_threshold=0.5):
        """
        Write one PSV per patient:
          PredictedProbability|PredictedLabel
        """
        os.makedirs(out_dir, exist_ok=True)

        if not hasattr(dataset, "files"):
            print("[WARN] dataset has no `.files`; cannot write per-patient PSV.")
            return

        files = list(dataset.files)

        starts = None
        total_raw = None

        if hasattr(dataset, "file_offsets"):
            starts = list(dataset.file_offsets)
        if hasattr(dataset, "total_raw"):
            total_raw = int(dataset.total_raw)

        n_files = len(files)

        if starts is not None and len(starts) == n_files:
            if total_raw is None:
                total_raw = int(len(prob_pos))
            ends = starts[1:] + [total_raw]
        elif starts is not None and len(starts) == n_files + 1:
            ends = starts[1:]
            starts = starts[:-1]
        else:
            print("[WARN] Cannot derive patient boundaries; missing/invalid file_offsets.")
            return

        prob_pos = np.asarray(prob_pos).reshape(-1)

        for fi, p in enumerate(files):
            s = int(starts[fi])
            e = int(ends[fi])
            patient_probs = prob_pos[s:e]

            patient_name = os.path.basename(p)
            out_path = os.path.join(out_dir, patient_name)

            pred_label = (patient_probs >= float(default_threshold)).astype(int)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("PredictedProbability|PredictedLabel\n")
                for pr, lb in zip(patient_probs, pred_label):
                    f.write(f"{float(pr):.6f}|{int(lb)}\n")

    def test(self, setting, test=0, flag="TEST", override_test_dir=None, test_tag=None):
        """
        test=1  -> load checkpoint from ./checkpoints/<setting>/checkpoint.pth
        override_test_dir -> temporarily uses this folder (relative to root_path) as test_dir
        test_tag -> name suffix for output folders (e.g. TEST / TRAIN_THRESH)
        """
        old_test_dir = None
        if override_test_dir is not None:
            old_test_dir = getattr(self.args, "test_dir", None)
            self.args.test_dir = override_test_dir

        test_data, test_loader = self._get_data(flag=flag)

        if test:
            print('loading model')
            ckpt = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        if override_test_dir is not None:
            self.args.test_dir = old_test_dir

        results_dir = os.path.join('./results', setting)
        os.makedirs(results_dir, exist_ok=True)

        if test_tag is None:
            test_tag = "TEST" if override_test_dir is None else "EXTRA"

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

        logits = torch.cat(all_logits, dim=0)
        trues = torch.cat(all_true, dim=0)
        print('test shape:', logits.shape, trues.shape)

        if logits.dim() >= 2 and logits.shape[-1] == 1:
            prob_pos = torch.sigmoid(logits).view(-1)
            pred_t = (prob_pos >= 0.5).long()
            true_np = trues.view(-1).long().numpy()
            conf = torch.where(pred_t == 1, prob_pos, 1.0 - prob_pos)

            probs_np = prob_pos.numpy().reshape(-1, 1)
            logits_np = logits.numpy()
            pred_np = pred_t.numpy()
            conf_np = conf.numpy()
        else:
            probs = F.softmax(logits, dim=-1)
            pred_t = torch.argmax(probs, dim=-1)
            true_np = trues.view(-1).long().numpy()

            logits_np = logits.numpy()
            probs_np = probs.numpy()
            pred_np = pred_t.numpy()
            conf_np = probs.max(dim=-1).values.numpy()
            prob_pos = probs[:, 1] if probs.shape[-1] == 2 else None

        np.save(os.path.join(results_dir, f"logits_{test_tag}.npy"), logits_np)
        np.save(os.path.join(results_dir, f"probs_{test_tag}.npy"), probs_np)
        np.save(os.path.join(results_dir, f"pred_{test_tag}.npy"), pred_np)
        np.save(os.path.join(results_dir, f"true_{test_tag}.npy"), true_np)

        out_df = {
            "sample_idx": np.arange(len(pred_np)),
            "true": true_np,
            "pred": pred_np,
            "conf": conf_np,
        }
        if prob_pos is not None:
            out_df["prob_pos"] = prob_pos.detach().cpu().numpy()
        pd.DataFrame(out_df).to_csv(os.path.join(results_dir, f"predictions_{test_tag}.csv"), index=False)

        accuracy = cal_accuracy(pred_np, true_np)
        print('accuracy:{}'.format(accuracy))

        # per-patient PSV for SepsisPSV style eval
        pred_psv_dir = os.path.join(results_dir, f"pred_psv_{test_tag}")
        if prob_pos is None and (logits.dim() >= 2 and logits.shape[-1] == 1):
            prob_pos_np = probs_np.reshape(-1)
        elif prob_pos is not None:
            prob_pos_np = prob_pos.detach().cpu().numpy()
        else:
            prob_pos_np = None

        if prob_pos_np is not None:
            self._write_sepsis_pred_psv_dir(test_data, prob_pos_np, pred_psv_dir, default_threshold=0.5)
            print("Wrote per-patient PSV predictions to:", pred_psv_dir)
        else:
            print("[INFO] Not writing per-patient PSV (non-binary output).")

        file_name = 'result_classification.txt'
        with open(os.path.join(results_dir, file_name), 'a', encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write(f"{test_tag} accuracy:{accuracy}\n\n")

        return
