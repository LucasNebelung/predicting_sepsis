import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, TemporalEmbedding
from torch import Tensor
from typing import Optional
from collections import namedtuple

# ============================================================
# Temporal Fusion Transformer (TFT)
# MODIFIED to support SepsisPSV *classification* in your pipeline:
#   - called as: model(batch_x, padding_mask, None, None)
#   - returns logits shape (B, 1)  -> Exp_Classification uses BCEWithLogitsLoss
#   - ONLY this file is changed (per your instruction)
# ============================================================

# static: time-independent features
# observed: time-varying features (past only, incl. ICULOS and engineered features like recency_*)
TypePos = namedtuple("TypePos", ["static", "observed"])

# Default datasets (kept)
datatype_dict = {
    "ETTh1": TypePos([], [x for x in range(7)]),
    "ETTm1": TypePos([], [x for x in range(7)]),
    # SepsisCSV and PSV is handled dynamically below (see _ensure_sepsis_mapping)
}


# -----------------------------
# Helpers
# -----------------------------
def get_known_len(embed_type, freq):
    if embed_type != "timeF":
        if freq == "t":
            return 5
        else:
            return 4
    else:
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        return freq_map[freq]


def _pool_last_valid(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
    """
    x:    (B, T, D)
    mask: (B, T) with 1.0 valid, 0.0 pad  (your SepsisPSVWindowDataset format)
    returns: (B, D) at last valid timestep per batch
    """
    # lengths = number of valid steps
    lengths = mask_1d.sum(dim=1).long().clamp(min=1)  # (B,)
    idx = (lengths - 1).view(-1, 1, 1)  # (B,1,1)
    idx = idx.expand(-1, 1, x.size(-1))  # (B,1,D)
    return x.gather(dim=1, index=idx).squeeze(1)  # (B,D)


def _ensure_sepsis_mapping(configs):
    """
    Creates datatype_dict['SepsisPSV'] or datatype_dict['SepsisCSV'] if missing.

    SepsisPSV:
      - assumes PhysioNet PSV base order and uses known static positions

    SepsisCSV:
      - makes NO column-order assumptions
      - treats ALL enc_in features as observed (static = [])
    """
    data_name = getattr(configs, "data", None)

    if data_name not in ("SepsisPSV", "SepsisCSV"):
        return

    if data_name in datatype_dict:
        return

    enc_in = int(getattr(configs, "enc_in", 0))
    if enc_in <= 0:
        raise ValueError(f"TFT({data_name}): configs.enc_in must be set before model init.")

    # --- CSV: safest generic behavior ---
    if data_name == "SepsisCSV":
        datatype_dict["SepsisCSV"] = TypePos(static=[], observed=list(range(enc_in)))
        return

    # --- PSV: keep your existing PhysioNet assumption ---
    base_dim = int(getattr(configs, "sepsis_base_dim", 40))

    if enc_in < base_dim:
        datatype_dict["SepsisPSV"] = TypePos(static=[], observed=list(range(enc_in)))
        return

    static_pos = [34, 35, 36, 37, 38]  # Age, Gender, Unit1, Unit2, HospAdmTime
    iculos_pos = 39

    observed_pos = [i for i in range(base_dim) if i not in static_pos]
    if iculos_pos not in observed_pos:
        observed_pos.append(iculos_pos)

    if enc_in > base_dim:
        observed_pos.extend(list(range(base_dim, enc_in)))

    datatype_dict["SepsisPSV"] = TypePos(static=static_pos, observed=observed_pos)

# -----------------------------
# Embeddings (kept for forecasting; extended for classification)
# -----------------------------
class TFTTemporalEmbedding(TemporalEmbedding):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TFTTemporalEmbedding, self).__init__(d_model, embed_type, freq)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        embedding_x = (
            torch.stack([month_x, day_x, weekday_x, hour_x, minute_x], dim=-2)
            if hasattr(self, "minute_embed")
            else torch.stack([month_x, day_x, weekday_x, hour_x], dim=-2)
        )
        return embedding_x


class TFTTimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TFTTimeFeatureEmbedding, self).__init__()
        d_inp = get_known_len(embed_type, freq)
        self.embed = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(d_inp)])

    def forward(self, x):
        return torch.stack([embed(x[:, :, i].unsqueeze(-1)) for i, embed in enumerate(self.embed)], dim=-2)


class TFTEmbedding(nn.Module):
    """
    Original TFT embedding (forecasting).
    For Sepsis classification in your pipeline we will NOT use known inputs,
    but we keep this class for compatibility with forecasting.
    """

    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.pred_len = configs.pred_len
        self.static_pos = datatype_dict[configs.data].static
        self.observed_pos = datatype_dict[configs.data].observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        self.static_embedding = (
            nn.ModuleList([DataEmbedding(1, configs.d_model, dropout=configs.dropout) for _ in range(self.static_len)])
            if self.static_len
            else None
        )
        self.observed_embedding = nn.ModuleList(
            [DataEmbedding(1, configs.d_model, dropout=configs.dropout) for _ in range(self.observed_len)]
        )
        self.known_embedding = (
            TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq)
            if configs.embed != "timeF"
            else TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.static_len:
            static_input = torch.stack(
                [
                    embed(x_enc[:, :1, self.static_pos[i]].unsqueeze(-1), None).squeeze(1)
                    for i, embed in enumerate(self.static_embedding)
                ],
                dim=-2,
            )  # [B,C,d]
        else:
            static_input = None

        observed_input = torch.stack(
            [embed(x_enc[:, :, self.observed_pos[i]].unsqueeze(-1), None) for i, embed in enumerate(self.observed_embedding)],
            dim=-2,
        )  # [B,T,C,d]

        x_mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len :, :]], dim=-2)
        known_input = self.known_embedding(x_mark)  # [B,T,C,d]

        return static_input, observed_input, known_input


# -----------------------------
# Core TFT blocks (unchanged)
# -----------------------------
class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GateAddNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(GateAddNorm, self).__init__()
        self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        x = self.glu(x)
        x = x + skip_a
        return self.layer_norm(self.projection(x))


class GRN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0):
        super(VariableSelectionNetwork, self).__init__()
        self.joint_grn = GRN(d_model * variable_num, variable_num, hidden_size=d_model, context_size=d_model, dropout=dropout)
        self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(variable_num)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        x_flattened = torch.flatten(x, start_dim=-2)
        selection_weights = self.joint_grn(x_flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)

        x_processed = torch.stack([grn(x[..., i, :]) for i, grn in enumerate(self.variable_grns)], dim=-1)
        selection_result = torch.matmul(x_processed, selection_weights.unsqueeze(-1)).squeeze(-1)
        return selection_result


class StaticCovariateEncoder(nn.Module):
    def __init__(self, d_model, static_len, dropout=0.0):
        super(StaticCovariateEncoder, self).__init__()
        self.static_vsn = VariableSelectionNetwork(d_model, static_len) if static_len else None
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(4)])

    def forward(self, static_input):
        if static_input is not None:
            static_features = self.static_vsn(static_input)
            return [grn(static_features) for grn in self.grns]
        else:
            return [None] * 4


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_heads = configs.n_heads
        assert configs.d_model % configs.n_heads == 0
        self.d_head = configs.d_model // configs.n_heads
        self.qkv_linears = nn.Linear(configs.d_model, (2 * self.n_heads + 1) * self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5

        # for classification we only need causal attention on seq_len
        example_len = max(1, int(getattr(configs, "seq_len", 1)) + int(getattr(configs, "pred_len", 0)))
        self.register_buffer("mask", torch.triu(torch.full((example_len, example_len), float("-inf")), 1))

    def forward(self, x):
        B, T, d_model = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.d_head)

        attn = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))  # [B,n,T,T]
        attn.mul_(self.scale)

        # if T differs from stored example_len, slice mask
        if self.mask.size(0) >= T:
            attn = attn + self.mask[:T, :T]
        else:
            # fallback: build mask on the fly
            dyn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
            attn = attn + dyn_mask

        prob = F.softmax(attn, dim=3)  # [B,n,T,T]
        out = torch.matmul(prob, v.unsqueeze(1))  # [B,n,T,d]
        out = torch.mean(out, dim=1)  # [B,T,d]
        out = self.out_projection(out)
        out = self.out_dropout(out)
        return out


# -----------------------------
# Classification-only encoder head (NEW)
# -----------------------------
class TFTClassificationEncoder(nn.Module):
    """
    A minimal TFT-style temporal encoder for classification:
      VSN -> LSTM -> GateAddNorm -> Attention -> GRN -> GateAddNorm
    Produces per-timestep features (B,T,d_model).
    """

    def __init__(self, configs):
        super().__init__()
        self.lstm = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model)
        self.attn = InterpretableMultiHeadAttention(configs)
        self.gate_after_attn = GateAddNorm(configs.d_model, configs.d_model)
        self.pw_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model)

    def forward(self, x: torch.Tensor):
        # x: (B,T,d)
        lstm_out, _ = self.lstm(x)  # (B,T,d)
        x = self.gate_after_lstm(lstm_out, x)

        attn_out = self.attn(x)
        x2 = self.gate_after_attn(attn_out, x)

        pw = self.pw_grn(x2)
        out = self.gate_final(pw, x2)
        return out  # (B,T,d)


# -----------------------------
# Main Model
# -----------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.pred_len = getattr(configs, "pred_len", 0)

        # Ensure Sepsis mapping exists before we read datatype_dict
        _ensure_sepsis_mapping(configs)

        # Number of variables (after mapping)
        self.static_pos = datatype_dict[configs.data].static
        self.observed_pos = datatype_dict[configs.data].observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        # ---- Forecasting modules (kept, unchanged) ----
        self.known_len = get_known_len(configs.embed, configs.freq)
        self.embedding = TFTEmbedding(configs)
        self.static_encoder = StaticCovariateEncoder(configs.d_model, self.static_len, dropout=configs.dropout)
        self.history_vsn = VariableSelectionNetwork(configs.d_model, self.observed_len + self.known_len, dropout=configs.dropout)
        self.future_vsn = VariableSelectionNetwork(configs.d_model, self.known_len, dropout=configs.dropout)

        # The original decoder is NOT used for classification in your pipeline
        # because Exp_Classification sets pred_len=0.
        # (Keeping it does not hurt forecasting tasks.)
        self.temporal_fusion_decoder = None
        try:
            # lazy import / keep compatibility if someone still uses forecasting here
            from typing import cast  # noqa
            self.temporal_fusion_decoder = TemporalFusionDecoder(configs)  # type: ignore[name-defined]
        except Exception:
            self.temporal_fusion_decoder = None

        # ---- Classification modules (NEW) ----
        if self.task_name == "classification":
            # per-variable embeddings (no known/time marks)
            self.static_embedding_cls = (
                nn.ModuleList([DataEmbedding(1, configs.d_model, dropout=configs.dropout) for _ in range(self.static_len)])
                if self.static_len
                else None
            )
            self.observed_embedding_cls = nn.ModuleList(
                [DataEmbedding(1, configs.d_model, dropout=configs.dropout) for _ in range(self.observed_len)]
            )

            self.static_encoder_cls = StaticCovariateEncoder(configs.d_model, self.static_len, dropout=configs.dropout)
            self.vsn_cls = VariableSelectionNetwork(configs.d_model, self.observed_len, dropout=configs.dropout)
            self.temporal_encoder_cls = TFTClassificationEncoder(configs)

            # Option B you chose: single logit (B,1)
            self.classifier = nn.Linear(configs.d_model, 1)

    # -----------------------------
    # Forecasting (kept)
    # -----------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        static_input, observed_input, known_input = self.embedding(x_enc, x_mark_enc, x_dec, x_mark_dec)
        c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        history_input = torch.cat([observed_input, known_input[:, : self.seq_len]], dim=-2)
        future_input = known_input[:, self.seq_len :]
        history_input = self.history_vsn(history_input, c_s)
        future_input = self.future_vsn(future_input, c_s)

        if self.temporal_fusion_decoder is None:
            raise RuntimeError("TFT: temporal_fusion_decoder not available.")

        dec_out = self.temporal_fusion_decoder(history_input, future_input, c_c, c_h, c_e)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # -----------------------------
    # Classification (NEW, for your pipeline)
    # -----------------------------
    def classification(self, x_enc: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        x_enc:        (B, T, F)
        padding_mask: (B, T) with 1 valid, 0 pad   (your Sepsis loader format)
        returns logits: (B, 1)
        """
        # Embed static vars from first timestep (they repeat anyway)
        if self.static_len:
            static_input = torch.stack(
                [
                    emb(x_enc[:, :1, self.static_pos[i]].unsqueeze(-1), None).squeeze(1)
                    for i, emb in enumerate(self.static_embedding_cls)
                ],
                dim=-2,
            )  # (B,C,d)
        else:
            static_input = None

        # Embed observed vars across time
        observed_input = torch.stack(
            [emb(x_enc[:, :, self.observed_pos[i]].unsqueeze(-1), None) for i, emb in enumerate(self.observed_embedding_cls)],
            dim=-2,
        )  # (B,T,C,d)

        # Static context
        c_s, _, _, _ = self.static_encoder_cls(static_input)  # c_s: (B,d) or None

        # Variable selection over observed variables
        x = self.vsn_cls(observed_input, c_s)  # (B,T,d)

        # Temporal encoding
        x = self.temporal_encoder_cls(x)  # (B,T,d)

        # Pool last valid timestep
        last = _pool_last_valid(x, padding_mask)  # (B,d)

        # Logit
        return self.classifier(last)  # (B,1)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Your pipeline calls: model(batch_x, padding_mask, None, None)
        # => x_mark_enc is actually padding_mask here
        if self.task_name == "classification":
            padding_mask = x_mark_enc
            return self.classification(x_enc, padding_mask)

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,pred_len,C]
            dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)
            return dec_out  # [B, T, D]

        return None


# ------------------------------------------------------------
# Original decoder class is referenced above (forecasting path)
# Keeping it here unchanged (from your file).
# ------------------------------------------------------------
class TemporalFusionDecoder(nn.Module):
    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.pred_len = configs.pred_len

        self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.future_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model)
        self.enrichment_grn = GRN(configs.d_model, configs.d_model, context_size=configs.d_model, dropout=configs.dropout)
        self.attention = InterpretableMultiHeadAttention(configs)
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model)
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, history_input, future_input, c_c, c_h, c_e):
        c = (c_c.unsqueeze(0), c_h.unsqueeze(0)) if c_c is not None and c_h is not None else None
        historical_features, state = self.history_encoder(history_input, c)
        future_features, _ = self.future_encoder(future_input, state)

        temporal_input = torch.cat([history_input, future_input], dim=1)
        temporal_features = torch.cat([historical_features, future_features], dim=1)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input)

        enriched_features = self.enrichment_grn(temporal_features, c_e)

        attention_out = self.attention(enriched_features)
        attention_out = self.gate_after_attention(attention_out[:, -self.pred_len :], enriched_features[:, -self.pred_len :])

        out = self.position_wise_grn(attention_out)
        out = self.gate_final(out, temporal_features[:, -self.pred_len :])
        return self.out_projection(out)
