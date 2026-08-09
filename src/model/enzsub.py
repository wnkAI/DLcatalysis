"""
DLCatalysis 2.0 — Three-branch residual prediction with dynamic gating.

Architecture:
    pred = y_seq + g * (y_sub + y_int)

    y_seq:  enzyme baseline activity (from ProtT5 mean-pooled)
    y_sub:  substrate effect (from GNN graph embedding)
    y_int:  enzyme-substrate interaction (from residue-atom cross-attention)
    g:      sample-wise dynamic gate (learned, no group statistics needed at inference)

Loss:
    L = L_regression + lambda_rank * L_ranking

    L_regression: LogCosh / Huber / MSE (configurable)
    L_ranking:    pairwise margin ranking loss for same-enzyme substrate pairs
"""
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import spearmanr

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from model.module import MLP
from util.featurize.seq_esm2 import esm_embedding


class EnzSub(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({'config': config})

        self.config       = config
        self.hidden_dim   = config['model']['hidden_dim']
        self.dropout_rate = config['model'].get('dropout', 0.2)

        self.automatic_optimization = True
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Metrics storage
        self.train_preds   = []; self.train_targets  = []
        self.val_preds     = []; self.val_targets    = []
        self.test_preds    = []; self.test_targets   = []
        # Ranking metrics accumulators (per epoch)
        self._rank_stats = {'train': [], 'val': []}
        # Final test metrics (populated in on_test_epoch_end)
        self._last_test_metrics = None

        self.log_dir = config['train']['log_path']
        os.makedirs(self.log_dir, exist_ok=True)
        model_name = config['model'].get('model_name', 'model')
        self.results_file = os.path.join(self.log_dir, f'{model_name}_{self.run_id}.csv')

        # ── Sequence encoder (ProtT5 frozen, precomputed) ──────────────
        self.unfreeze_schedule = config['model'].get('unfreeze_schedule', {})
        self.current_strategy = None

        init_device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
        encoder_config = dict(config['model'])
        encoder_config['precomputed_only'] = True
        self.esm_encoder = esm_embedding(device=init_device, config=encoder_config)

        for param in self.esm_encoder.parameters():
            param.requires_grad = False

        # ── Sequence projection ───────────────────────────────────────
        seq_cfg = config['model']['seq_module']
        esm_dim = self.esm_encoder.embedding_dim
        self.seq_mlp = nn.Sequential(
            nn.Linear(esm_dim, seq_cfg['seq_hidden_dim']),
            nn.LayerNorm(seq_cfg['seq_hidden_dim']),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(seq_cfg['seq_hidden_dim'], self.hidden_dim)
        )

        # ── Attention pooling ─────────────────────────────────────────
        self.attn_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

        # ── Substrate projections (Morgan/MolT5/Grover) ──────────────
        self.use_morgan = config['model']['embedding'].get('morgan', False)
        self.use_molt5 = config['model']['embedding'].get('molt5', False)
        self.use_grover = config['model']['embedding'].get('grover', False)
        feature_norm = config['model'].get('feature_norm', True)

        if self.use_morgan:
            morgan_dim = config['model']['embedding'].get('morgan_dim', 1024)
            self.morgan_proj = nn.Sequential(
                nn.Linear(morgan_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim) if feature_norm else nn.Identity(),
                nn.Dropout(self.dropout_rate)
            )
        if self.use_molt5:
            molt5_dim = config['model']['embedding'].get('molt5_dim', 768)
            self.molt5_proj = nn.Sequential(
                nn.Linear(molt5_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim) if feature_norm else nn.Identity(),
                nn.Dropout(self.dropout_rate)
            )
        if self.use_grover:
            grover_dim = config['model']['embedding'].get('grover_dim', 4885)
            self.grover_proj = nn.Sequential(
                nn.Linear(grover_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim) if feature_norm else nn.Identity(),
                nn.Dropout(self.dropout_rate)
            )

        # ── EC number embedding ───────────────────────────────────────
        self.use_ec = config['model'].get('use_ec', False)
        if self.use_ec:
            ec_cfg = config['model'].get('ec', {})
            ec_embed_dim = ec_cfg.get('ec_embed_dim', 16)
            self.ec1_emb = nn.Embedding(ec_cfg.get('ec1_vocab', 8), ec_embed_dim, padding_idx=0)
            self.ec2_emb = nn.Embedding(ec_cfg.get('ec2_vocab', 200), ec_embed_dim, padding_idx=0)
            self.ec3_emb = nn.Embedding(ec_cfg.get('ec3_vocab', 200), ec_embed_dim, padding_idx=0)
            self.ec4_emb = nn.Embedding(ec_cfg.get('ec4_vocab', 1200), ec_embed_dim, padding_idx=0)
            self.ec_proj = nn.Sequential(
                nn.Linear(4 * ec_embed_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim), nn.LeakyReLU(), nn.Dropout(self.dropout_rate),
            )

        # ── Cross-attention (residue-atom level) ──────────────────────
        attn_cfg = config['model'].get('cross_attention', {})
        n_head = attn_cfg.get('n_head', 4)
        n_cross_layers = attn_cfg.get('n_layers', 2)

        self.cross_layers = nn.ModuleList()
        for _ in range(n_cross_layers):
            self.cross_layers.append(nn.ModuleDict({
                'sub_self_attn': nn.MultiheadAttention(self.hidden_dim, n_head, batch_first=True, dropout=self.dropout_rate),
                'sub_self_norm': nn.LayerNorm(self.hidden_dim),
                'enz2sub': nn.MultiheadAttention(self.hidden_dim, n_head, batch_first=True, dropout=self.dropout_rate),
                'enz2sub_norm': nn.LayerNorm(self.hidden_dim),
                'sub2enz': nn.MultiheadAttention(self.hidden_dim, n_head, batch_first=True, dropout=self.dropout_rate),
                'sub2enz_norm': nn.LayerNorm(self.hidden_dim),
            }))

        # ── Three-branch prediction heads ─────────────────────────────
        hdr = config['model']['output_header']

        # Branch 1: enzyme baseline (only enzyme info)
        self.head_seq = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=hdr['hidden_dim'],
            num_layer=hdr['num_layers'],
            norm=hdr['norm_fn'],
            act_fn=hdr['act_fn'],
            dropout=self.dropout_rate
        )

        # Branch 2: substrate effect (pure substrate, no EC)
        self.head_sub = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=hdr['hidden_dim'],
            num_layer=hdr['num_layers'],
            norm=hdr['norm_fn'],
            act_fn=hdr['act_fn'],
            dropout=self.dropout_rate
        )

        # Branch 3: interaction (cross-attention + EC for reaction context)
        int_in_dim = self.hidden_dim * 2  # enzyme_attended + substrate_attended
        if self.use_ec:
            int_in_dim += self.hidden_dim  # + EC embedding
        self.head_int = MLP(
            in_dim=int_in_dim,
            out_dim=1,
            hidden_dim=hdr['hidden_dim'],
            num_layer=hdr['num_layers'],
            norm=hdr['norm_fn'],
            act_fn=hdr['act_fn'],
            dropout=self.dropout_rate
        )

        # Gate: EC defines reaction prerequisites (compatibility filter)
        gate_in_dim = self.hidden_dim * 2  # enzyme_pooled + graph_emb
        if self.use_ec:
            gate_in_dim += self.hidden_dim  # + EC embedding
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────
    def _init_weights(self):
        esm_module_ids = set(id(m) for m in self.esm_encoder.modules())
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in esm_module_ids:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Attention pooling (safe for all-padding) ──────────────────────
    def _attn_pool_seq(self, feat, mask):
        """
        feat: (B, L, D), mask: (B, L, 1) float, 1=valid 0=padding
        """
        scores = self.attn_pool(feat)  # (B, L, 1)
        scores = scores.masked_fill(mask < 0.5, torch.finfo(scores.dtype).min)
        has_valid = (mask > 0.5).any(dim=1, keepdim=True)  # (B, 1, 1)
        weights = torch.softmax(scores, dim=1)
        weights = weights * has_valid.float()
        return (feat * weights).sum(dim=1)  # (B, D)

    # ── Encode enzyme ─────────────────────────────────────────────────
    def _encode_enzyme(self, G, batch_size):
        if not hasattr(G, 'SEQ_seq_padding_mask'):
            enzyme_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
            enzyme_pooled = torch.zeros(batch_size, self.hidden_dim, device=self.device)
            seq_mask = torch.ones(batch_size, 1, 1, device=self.device)
            return enzyme_feat, enzyme_pooled, seq_mask

        max_len = G.SEQ_seq_padding_mask.shape[1]
        seq_mask = (~G.SEQ_seq_padding_mask).unsqueeze(-1).float().to(self.device)

        x_pro_seq = G.SEQ_embedding.view(batch_size, max_len, -1).to(self.device).float()

        enzyme_feat = self.seq_mlp(x_pro_seq)  # (B, L, hidden_dim)

        enzyme_pooled = self._attn_pool_seq(enzyme_feat, seq_mask)  # (B, hidden_dim)
        return enzyme_feat, enzyme_pooled, seq_mask

    # ── Encode substrate (Morgan/MolT5/Grover) ────────────────────────
    def _encode_substrate(self, G, batch_size):
        """Encode substrate using legacy fixed features."""
        features = []
        _zero = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        if self.use_morgan:
            if hasattr(G, 'MOL_morgan'):
                feat = self.morgan_proj(G.MOL_morgan.to(self.device).view(batch_size, -1))
            else:
                feat = _zero
            features.append(feat)

        if self.use_molt5:
            if hasattr(G, 'MOL_molt5'):
                feat = self.molt5_proj(G.MOL_molt5.to(self.device).view(batch_size, -1))
            else:
                feat = _zero
            features.append(feat)

        if self.use_grover:
            if hasattr(G, 'MOL_grover_mean'):
                feat = self.grover_proj(G.MOL_grover_mean.to(self.device).view(batch_size, -1))
            else:
                feat = _zero
            features.append(feat)

        if features:
            # Stack as token sequence for cross-attention compatibility
            sub_tokens = torch.stack(features, dim=1)  # (B, N_mods, hidden_dim)
            sub_pooled = sub_tokens.mean(dim=1)  # (B, hidden_dim)
            sub_mask = torch.ones(batch_size, len(features), dtype=torch.bool, device=self.device)
            return sub_tokens, sub_pooled, sub_mask

        return None, None, None

    # ── EC encoding ───────────────────────────────────────────────────
    def _encode_ec(self, G, batch_size):
        if not (self.use_ec and hasattr(G, 'EC_ids')):
            return None
        ec = G.EC_ids.to(self.device)
        if ec.dim() == 1:
            ec = ec.view(batch_size, 4)
        ec_cat = torch.cat([
            self.ec1_emb(ec[:, 0]), self.ec2_emb(ec[:, 1]),
            self.ec3_emb(ec[:, 2]), self.ec4_emb(ec[:, 3]),
        ], dim=-1)
        return self.ec_proj(ec_cat)

    # ── Cross-attention (multi-layer) ─────────────────────────────────
    def _cross_attend(self, enzyme_feat, sub_tokens, seq_mask, sub_mask):
        """
        enzyme_feat: (B, L_seq, D)
        sub_tokens:  (B, L_sub, D)
        seq_mask:    (B, L_seq, 1) float
        sub_mask:    (B, L_sub) bool, True=valid
        """
        enz = enzyme_feat
        sub = sub_tokens

        # Padding masks for attention (True = ignore)
        enz_pad_mask = (seq_mask.squeeze(-1) < 0.5)  # (B, L_seq)
        sub_pad_mask = ~sub_mask  # (B, L_sub)

        for layer in self.cross_layers:
            # Substrate self-attention (skip if all padding)
            if not sub_pad_mask.all():
                sub_self, _ = layer['sub_self_attn'](sub, sub, sub, key_padding_mask=sub_pad_mask)
                sub = layer['sub_self_norm'](sub + sub_self)

            # Enzyme -> Substrate (skip if substrate all padding)
            if not sub_pad_mask.all():
                enz_att, _ = layer['enz2sub'](enz, sub, sub, key_padding_mask=sub_pad_mask)
                enz = layer['enz2sub_norm'](enz + enz_att)

            # Substrate -> Enzyme (skip if enzyme all padding)
            if not enz_pad_mask.all():
                sub_att, _ = layer['sub2enz'](sub, enz, enz, key_padding_mask=enz_pad_mask)
                sub = layer['sub2enz_norm'](sub + sub_att)

        # Pool attended representations
        enz_attended = self._attn_pool_seq(enz, seq_mask)  # (B, D)

        # Pool substrate (masked mean)
        sub_mask_float = sub_mask.unsqueeze(-1).float()  # (B, L_sub, 1)
        sub_attended = (sub * sub_mask_float).sum(dim=1) / sub_mask_float.sum(dim=1).clamp(min=1e-8)  # (B, D)

        return enz_attended, sub_attended

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, G):
        batch_size = G.num_graphs if hasattr(G, 'num_graphs') else 1

        if hasattr(G, 'y') and G.y.device != self.device:
            G.y = G.y.to(self.device)

        # 1) Encode enzyme
        enzyme_feat, enzyme_pooled, seq_mask = self._encode_enzyme(G, batch_size)

        # 2) Encode substrate
        atom_tokens, graph_emb, atom_mask = self._encode_substrate(G, batch_size)

        # 3) EC encoding
        ec_feat = self._encode_ec(G, batch_size)

        # 4) Cross-attention (residue-atom interaction)
        cross_disabled = self.config['model'].get('cross_attention', {}).get('disable', False)
        if not cross_disabled and atom_tokens is not None and atom_mask is not None:
            enz_attended, sub_attended = self._cross_attend(
                enzyme_feat, atom_tokens, seq_mask, atom_mask
            )
        else:
            enz_attended = enzyme_pooled
            sub_attended = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # 5) Three-branch prediction
        # EC zero fallback for concat
        _ec = ec_feat if ec_feat is not None else torch.zeros(batch_size, self.hidden_dim, device=self.device)
        _graph = graph_emb if graph_emb is not None else torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # Branch 1: enzyme baseline (enzyme only)
        y_seq = self.head_seq(enzyme_pooled)  # (B, 1)

        # Branch 2: substrate effect (pure substrate, no EC)
        y_sub = self.head_sub(_graph)  # (B, 1)

        # Branch 3: interaction (cross-attention + EC for reaction context)
        int_input = torch.cat([enz_attended, sub_attended], dim=-1)
        if self.use_ec:
            int_input = torch.cat([int_input, _ec], dim=-1)
        y_int = self.head_int(int_input)  # (B, 1)

        # 6) Dynamic gate: EC as reaction compatibility filter
        gate_input = torch.cat([enzyme_pooled, _graph], dim=-1)
        if self.use_ec:
            gate_input = torch.cat([gate_input, _ec], dim=-1)
        g = self.gate_net(gate_input)  # (B, 1), range [0, 1]

        # 7) Final prediction: y_seq always preserved, gate scales delta
        delta = y_sub + y_int
        y_pred = y_seq + g * delta

        y_true = G.y
        return y_pred, y_true

    # ── Loss ──────────────────────────────────────────────────────────
    def _prepare_target(self, y_true, stage):
        y_target = torch.log10(y_true.float().clamp(min=1e-8))
        if stage == 'train':
            label_noise = self.config['model'].get('label_noise', 0.0)
            if label_noise > 0:
                y_target = y_target + torch.randn_like(y_target) * label_noise
        return y_target

    def _pairwise_ranking(self, pred, target, ids, margin, min_diff=0.1, max_pairs=16):
        """Generic pairwise ranking loss. Returns (loss, n_pairs, rank_acc)."""
        pred = pred.squeeze(-1)
        target = target.squeeze() if target.dim() > 1 else target

        loss = torch.tensor(0.0, device=self.device)
        n_pairs = 0
        n_correct = 0

        from collections import defaultdict
        groups = defaultdict(list)
        for i, gid in enumerate(ids):
            groups[gid].append(i)

        for gid, indices in groups.items():
            if len(indices) < 2:
                continue
            pairs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if (target[indices[i]] - target[indices[j]]).abs() >= min_diff:
                        pairs.append((indices[i], indices[j]))
            if not pairs:
                continue
            if len(pairs) > max_pairs:
                idx = torch.randperm(len(pairs))[:max_pairs].tolist()
                pairs = [pairs[k] for k in idx]
            for idx_i, idx_j in pairs:
                diff_true = target[idx_i] - target[idx_j]
                diff_pred = pred[idx_i] - pred[idx_j]
                sign = torch.sign(diff_true)
                loss = loss + F.relu(margin - sign * diff_pred)
                n_pairs += 1
                if diff_pred.sign() == sign:
                    n_correct += 1

        rank_acc = n_correct / max(n_pairs, 1)
        return loss / max(n_pairs, 1), n_pairs, rank_acc

    def _ranking_loss(self, y_pred, y_target, G):
        """Same-enzyme ranking: fixed enzyme, rank substrates."""
        if not hasattr(G, 'SEQ_seq_id'):
            return torch.tensor(0.0, device=self.device), 0, 0.0
        cfg = self.config['model']
        return self._pairwise_ranking(
            y_pred, y_target, G.SEQ_seq_id,
            margin=cfg.get('rank_margin_enzyme', 0.1),
            min_diff=cfg.get('rank_min_diff_enzyme', 0.1),
            max_pairs=cfg.get('rank_max_pairs_enzyme', 16),
        )

    def _ranking_loss_substrate(self, y_pred, y_target, G):
        """Same-substrate ranking: fixed substrate, rank enzymes."""
        if not hasattr(G, 'MOL_smi_id'):
            return torch.tensor(0.0, device=self.device), 0, 0.0
        cfg = self.config['model']
        return self._pairwise_ranking(
            y_pred, y_target, G.MOL_smi_id,
            margin=cfg.get('rank_margin_substrate', 0.1),
            min_diff=cfg.get('rank_min_diff_substrate', 0.1),
            max_pairs=cfg.get('rank_max_pairs_substrate', 16),
        )

    def get_loss(self, y_pred, y_true, stage, G=None):
        pred = y_pred.squeeze(-1).float()
        y_target = self._prepare_target(y_true, stage).float()

        # Base regression loss
        loss_type = self.config['model'].get('loss_type', 'mse')
        if loss_type == 'huber':
            delta = self.config['model'].get('huber_delta', 1.0)
            base_loss = F.huber_loss(pred, y_target, delta=delta)
        elif loss_type == 'logcosh':
            diff = pred - y_target
            base_loss = (diff + F.softplus(-2.0 * diff) - math.log(2.0)).mean()
        else:
            base_loss = F.mse_loss(pred, y_target)

        loss = base_loss

        # PCC loss (optional)
        pcc_weight = self.config['model'].get('pcc_loss_weight', 0.0)
        if pcc_weight > 0.0 and pred.shape[0] >= 16:
            vx = pred - pred.mean()
            vy = y_target - y_target.mean()
            pcc = (vx * vy).sum() / (vx.norm().clamp(min=1e-4) * vy.norm().clamp(min=1e-4))
            loss = loss + pcc_weight * (1.0 - pcc.clamp(-1.0, 1.0))

        # Ranking losses (training only) — uses clean target without label noise
        if stage == 'train' and G is not None:
            y_target_clean = torch.log10(y_true.float().clamp(min=1e-8))
            # Same-enzyme ranking: learn substrate sensitivity
            rank_weight = self.config['model'].get('rank_loss_weight', 0.0)
            if rank_weight > 0.0:
                rl_enz, np_enz, ra_enz = self._ranking_loss(y_pred, y_target_clean, G)
                loss = loss + rank_weight * rl_enz
                self.log('train/rank_loss_enzyme', rl_enz, prog_bar=False, logger=True, sync_dist=True)
                self.log('train/rank_pairs_enzyme', float(np_enz), prog_bar=False, logger=True, sync_dist=True)
                self.log('train/rank_acc_enzyme', ra_enz, prog_bar=False, logger=True, sync_dist=True)
            # Same-substrate ranking: learn enzyme sensitivity
            rank_sub_weight = self.config['model'].get('rank_loss_substrate_weight', 0.0)
            if rank_sub_weight > 0.0:
                rl_sub, np_sub, ra_sub = self._ranking_loss_substrate(y_pred, y_target_clean, G)
                loss = loss + rank_sub_weight * rl_sub
                self.log('train/rank_loss_substrate', rl_sub, prog_bar=False, logger=True, sync_dist=True)
                self.log('train/rank_pairs_substrate', float(np_sub), prog_bar=False, logger=True, sync_dist=True)
                self.log('train/rank_acc_substrate', ra_sub, prog_bar=False, logger=True, sync_dist=True)

        if stage is not None:
            self.log(f'{stage}_MSE', F.mse_loss(pred, y_target),
                     prog_bar=True, logger=True, sync_dist=True, batch_size=y_true.size(0))
        return loss

    # ── Metrics ───────────────────────────────────────────────────────
    def calculate_metrics(self, y_pred, y_true):
        y_pred_np = y_pred.cpu().numpy().flatten()
        y_true_np = y_true.cpu().numpy().flatten()
        y_true_log = np.log10(y_true_np + 1e-8)
        y_pred_log = y_pred_np

        r2 = r2_score(y_true_log, y_pred_log)
        pcc = np.corrcoef(y_true_log, y_pred_log)[0, 1]
        if np.isnan(pcc): pcc = 0.0
        scc, _ = spearmanr(y_true_log, y_pred_log)
        if np.isnan(scc): scc = 0.0
        mse = mean_squared_error(y_true_log, y_pred_log)
        mae = mean_absolute_error(y_true_log, y_pred_log)
        return {'R2': r2, 'PCC': pcc, 'SCC': scc, 'MSE': mse,
                'MAE': mae, 'RMSE': np.sqrt(mse)}

    # ── Training steps ────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        loss = self.get_loss(y_pred, y_true, 'train', G=batch)
        self.train_preds.append(y_pred.detach().cpu().float())
        self.train_targets.append(y_true.detach().cpu().float())
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        loss = self.get_loss(y_pred, y_true, 'val', G=batch)
        self.val_preds.append(y_pred.detach().cpu().float())
        self.val_targets.append(y_true.detach().cpu().float())
        # Accumulate ranking stats per batch
        with torch.no_grad():
            y_target_clean = torch.log10(y_true.float().clamp(min=1e-8))
            stats = {}
            if hasattr(batch, 'SEQ_seq_id'):
                _, np_enz, ra_enz = self._ranking_loss(y_pred, y_target_clean, batch)
                stats['rank_pairs_enz'] = np_enz
                stats['rank_acc_enz'] = ra_enz
            if hasattr(batch, 'MOL_smi_id'):
                _, np_sub, ra_sub = self._ranking_loss_substrate(y_pred, y_target_clean, batch)
                stats['rank_pairs_sub'] = np_sub
                stats['rank_acc_sub'] = ra_sub
            if stats:
                self._rank_stats['val'].append(stats)
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        self.test_preds.append(y_pred.detach().cpu().float())
        self.test_targets.append(y_true.detach().cpu().float())
        return y_pred, y_true

    # ── Epoch end ─────────────────────────────────────────────────────
    def _flush_epoch(self, split):
        preds = getattr(self, f'{split}_preds')
        targets = getattr(self, f'{split}_targets')
        if not preds:
            return
        metrics = self.calculate_metrics(torch.cat(preds), torch.cat(targets))
        # Always include fixed ranking columns (NaN when not available) to keep CSV schema stable
        for key in ['rank_pairs_enz', 'rank_acc_enz', 'rank_pairs_sub', 'rank_acc_sub']:
            metrics[key] = float('nan')
        if self._rank_stats.get(split):
            stats_list = self._rank_stats[split]
            for key in ['rank_pairs_enz', 'rank_acc_enz', 'rank_pairs_sub', 'rank_acc_sub']:
                vals = [s[key] for s in stats_list if key in s]
                if vals:
                    if 'pairs' in key:
                        metrics[key] = sum(vals)
                    else:
                        pair_key = key.replace('acc', 'pairs')
                        pairs = [s.get(pair_key, 1) for s in stats_list if key in s]
                        total = sum(pairs)
                        metrics[key] = sum(v * p for v, p in zip(vals, pairs)) / max(total, 1)
            self._rank_stats[split] = []
        self._save_results(split, metrics, len(torch.cat(targets)))
        setattr(self, f'{split}_preds', [])
        setattr(self, f'{split}_targets', [])

    def on_train_epoch_end(self):      self._flush_epoch('train')
    def on_validation_epoch_end(self): self._flush_epoch('val')

    def on_test_epoch_end(self):
        if not self.test_preds:
            return
        all_preds = torch.cat(self.test_preds)
        all_targets = torch.cat(self.test_targets)
        # DDP-aware: gather predictions from every rank before computing metrics.
        # Without this, _last_test_metrics would only reflect the local shard.
        if self.trainer is not None and self.trainer.world_size > 1:
            all_preds = self.all_gather(all_preds).reshape(-1)
            all_targets = self.all_gather(all_targets).reshape(-1)
        metrics = self.calculate_metrics(all_preds, all_targets)
        # Store final test metrics on the module for direct access by train() loop
        self._last_test_metrics = dict(metrics)
        # Keep CSV schema stable: add ranking columns as NaN for test split
        for key in ['rank_pairs_enz', 'rank_acc_enz', 'rank_pairs_sub', 'rank_acc_sub']:
            metrics[key] = float('nan')
        self._save_results('test', metrics, len(all_targets))
        print(f"\nTEST RESULTS -- {self.run_id}")
        for k in ['R2', 'PCC', 'SCC', 'MSE', 'MAE', 'RMSE']:
            print(f"  {k}: {metrics[k]:.6f}")
        self.test_preds = []
        self.test_targets = []

    def _save_results(self, dataset_name, metrics, n_samples):
        row = pd.DataFrame([{'epoch': self.current_epoch, 'split': dataset_name, **metrics, 'samples': n_samples}])
        row.to_csv(self.results_file, mode='a', header=not os.path.exists(self.results_file), index=False)
        for metric_name, metric_value in metrics.items():
            self.log(f'{dataset_name}/{metric_name}', metric_value,
                     prog_bar=metric_name in ['R2', 'PCC', 'MSE'],
                     logger=True, sync_dist=True)

    # ── Optimizer ─────────────────────────────────────────────────────
    def configure_optimizers(self):
        main_lr = float(self.config['train']['optimizer']['lr'])
        wd = float(self.config['train']['optimizer'].get('weight_decay', 0.01))

        # ESM encoder is frozen; only downstream modules train
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found.")

        param_groups = [{'params': trainable, 'lr': main_lr, 'weight_decay': wd}]

        optimizer = torch.optim.AdamW(param_groups)

        warm_epochs = int(self.config['train']['optimizer'].get('warm_epoch', 5))
        max_epochs = int(self.config['train']['max_epochs'])
        min_lr_ratio = float(self.config['train']['optimizer']['min_lr']) / max(main_lr, 1e-12)

        def lr_lambda(epoch):
            if epoch < warm_epochs:
                return (epoch + 1) / max(warm_epochs, 1)
            progress = (epoch - warm_epochs) / max(max_epochs - warm_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        }
