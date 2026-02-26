import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from chronos import BaseChronosPipeline


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.classifier = nn.LazyLinear(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        quantiles, dec_out = self.model.predict_quantiles(x_enc.cpu().numpy(), prediction_length=self.pred_len, quantile_levels=[0.1, 0.5, 0.9])
        dec_out = torch.stack(dec_out, dim=0).to(x_enc.device)
        dec_out= dec_out.permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'classification':
            means = x_enc.mean(1, keepdim=True).detach()
            x_norm = x_enc.sub(means)
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm.div(stdev)

            x_tokens = x_norm.permute(0, 2, 1)
            _, dec_out = self.model.predict_quantiles(
                x_tokens.detach().cpu().numpy(),
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
            )

            dec_out = torch.stack(dec_out, dim=0).to(device=x_enc.device, dtype=x_enc.dtype)
            # dec_out: (B, 1, C), pooled representation: (B, C)
            pooled_repr = dec_out[:, -1, :]
            logits = self.classifier(pooled_repr)
            return logits
        return None
