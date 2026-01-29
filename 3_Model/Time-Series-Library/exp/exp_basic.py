# exp/exp_basic.py
import os
import importlib
import torch


class Exp_Basic(object):
    """
    EXP_BASIC (LAZY MODEL IMPORT VERSION)

    Why this exists:
    - Your previous exp_basic.py imported *every* model at module import time:
      Crossformer, TimesFM, Chronos, Moirai, ... etc.
    - When PyTorch DataLoader spawns workers, each process re-imports Python modules.
      That caused the "Loaded PyTorch TimesFM..." message ~70 times and huge startup delay.

    What this does:
    - Only imports the single model specified by args.model
    - Keeps the same interface expected by Exp_Classification:
        self.model_dict[self.args.model].Model(self.args)
    """

    def __init__(self, args):
        self.args = args

        # Only keep the module for the requested model (lazy load)
        model_module = self._import_model_module(args.model)
        self.model_dict = {args.model: model_module}

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    # -----------------------------
    # model importing (lazy)
    # -----------------------------
    def _import_model_module(self, model_name: str):
        """
        Returns the imported module for the given model name.

        We assume the canonical TimesLib layout:
          models/<ModelName>.py  (e.g., models/Crossformer.py)
        and that each module defines class Model.

        Special-case: Mamba (kept from original behavior).
        """
        if model_name == "Mamba":
            print("Please make sure you have successfully installed mamba_ssm")
            return importlib.import_module("models.Mamba")

        # Most models match file name == model_name
        tried = []
        candidates = [
            f"models.{model_name}",               # e.g. models.TemporalFusionTransformer
        ]

        # Optional fallback for older naming conventions if you ever need it
        # (kept conservative: does not guess wildly)
        if model_name.endswith(".py"):
            candidates.append(f"models.{model_name[:-3]}")

        last_err = None
        for mod_path in candidates:
            try:
                return importlib.import_module(mod_path)
            except Exception as e:
                tried.append(mod_path)
                last_err = e

        raise ImportError(
            f"Could not import model '{model_name}'. Tried: {tried}. "
            f"Last error: {repr(last_err)}"
        )

    # -----------------------------
    # required hooks (implemented in subclasses)
    # -----------------------------
    def _build_model(self):
        raise NotImplementedError

    # -----------------------------
    # device
    # -----------------------------
    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # -----------------------------
    # placeholders
    # -----------------------------
    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
