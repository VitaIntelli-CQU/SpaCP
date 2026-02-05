from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from timm.models.vision_transformer import Mlp, SwiGLUPacked
from torch_geometric.utils import to_dense_batch

from utils import get_activation

__all__ = [
    "FrameAveraging",
    "GeneUpdate",
    "MLPAttnEdgeAggregation",
    "TransformerBlock",
    "SpatialTransformer",
]

class FrameAveraging(nn.Module):
    def __init__(self, dim=3, backward=False):
        super(FrameAveraging, self).__init__()

        self.dim = dim
        self.n_frames = 2 ** dim
        self.ops = self.create_ops(dim)  # [2^dim, dim]
        self.backward = backward

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim = -1)
        operations = rearrange(operations, '... d -> (...) d')
        return operations

    def create_frame(self, X, mask=None):
        assert X.shape[-1] == self.dim, f'expected points of dimension {self.dim}, but received {X.shape[-1]}'

        if mask is None:
            mask = torch.ones(*X.shape[:-1], device=X.device).bool()
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,dim]
        X_ = X.masked_fill(~mask, 0.)

        C = torch.bmm(X_.transpose(1,2), X_)  # [B,dim,dim] (Cov)
        if not self.backward:
            C = C.detach()

        _, eigenvectors = torch.linalg.eigh(C, UPLO='U')  # [B,dim,dim]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(X.device) * eigenvectors.unsqueeze(1)  # [1,2^dim,1,dim] x [B,1,dim,dim] -> [B,2^dim,dim,dim]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2,3), X)  # transpose is inverse [B,2^dim,N,dim]

        h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        if mask is None:
            return X
        return X * mask.unsqueeze(-1)

class GeneUpdate(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_genes,
            proj_drop=0.,
            non_negative=False,
        ):
        super(GeneUpdate, self).__init__()    
        self.non_negative = non_negative

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )
    
    def forward(self, features):
        update = self.output(features) 
        if self.non_negative:
            update = F.relu(update)
        return update

class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
            self, 
            d_model, 
            d_edge_model,
            n_genes,          
            d_attr=0,         
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            activation='gelu',
        ):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)
        
        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads
        self.n_genes = n_genes
        self.d_attr = d_attr

        self.mlp_attn_in_dim = self.d_head * 2 + self.d_edge_head + self.n_genes + self.d_attr

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.mlp_attn_in_dim,
                hidden_features=d_model, 
                out_features=1, 
                drop=proj_drop, 
                norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim+1, hidden_features=d_edge_model, 
                out_features=d_edge_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model+d_edge_model, hidden_features=d_model, 
                out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.mlp_attn_in_dim,
                hidden_features=d_model, 
                out_features=1, 
                drop=proj_drop, 
                norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim+1, hidden_features=d_edge_model, out_features=d_edge_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model+d_edge_model, hidden_features=d_model, out_features=d_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None, attr_emb=None):
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        
        message_parts = [q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand]
        
        if attr_emb is not None and self.d_attr > 0:
            attr_emb_neighbors = attr_emb[neighbor_indices]
            attr_emb_self = attr_emb.unsqueeze(1)
            attr_diff = torch.abs(attr_emb_neighbors - attr_emb_self)
            attr_diff = attr_diff.unsqueeze(2).expand(n_tokens, n_neighbors, n_heads, -1)
            message_parts.append(attr_diff)
        
        message = torch.cat(message_parts, dim=-1)
        
        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))

class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            d_edge_model,
            n_genes,
            d_attr=0,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            gene_exp_non_negative=True,
            mlp_ratio=4.0,
        ):
        super(TransformerBlock, self).__init__()

        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, 
            d_edge_model=d_edge_model, 
            n_genes=n_genes,
            d_attr=d_attr,
            n_heads=n_heads, 
            proj_drop=proj_drop, 
            attn_drop=attn_drop, 
            activation=activation
        )

        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, attr_emb=None):
        context_token_embs = self.attn(gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None, attr_emb=attr_emb)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        gene_exp = self.gene_updater(token_embs)

        return gene_exp, token_embs

class SpatialTransformer(nn.Module):
    def __init__(self, config, kernel_scale=None):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors
        self.kernel_scale = kernel_scale.to(config.device) if kernel_scale is not None else None
        self.d_attr = getattr(config, "d_attr", 0)

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_edge_model, 
                          n_genes=config.n_genes, 
                          d_attr=self.d_attr,
                          n_heads=config.n_heads, 
                          activation=config.act, 
                          attn_drop=config.attn_dropout, 
                          proj_drop=config.dropout, 
                          gene_exp_non_negative=getattr(config, "gene_exp_non_negative", True)
                        ) \
                for i in range(config.n_layers)
        ])

    def _build_graph(self, coords, attr_idx, batch_idx, n_neighbors, exclude_self=True):
        """Build neighbor indices using an attr-based batch mask."""
        exclude_self_mask = torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)

        if self.kernel_scale is not None:
            scale = self.kernel_scale[attr_idx]
            assert scale.shape == coords.shape, f"scale shape {scale.shape} must match coords shape {coords.shape}"
            coords_scaled = coords * scale
        else:
            coords_scaled = coords

        rel_pos = rearrange(coords_scaled, 'n d -> n 1 d') - rearrange(coords_scaled, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim = -1).detach()
        
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim = -1, largest = False)
        return nearest_indices

    def forward(self, gene_exp, features, coords, attr_idx, attr_emb=None):
        """Run spatial transformer blocks with attr-conditioned neighbors."""
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device
        
        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells]
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]
        attr_idx_masked = attr_idx
        
        features = features[~pad_mask]
        coords = coords[~pad_mask]
        gene_exp = gene_exp[~pad_mask]
        if attr_emb is not None:
            attr_emb = attr_emb[~pad_mask]

        nearest_indices = self._build_graph(
            coords, attr_idx_masked, batch_idx, min(self.n_neighbors, N_cells), exclude_self=True
        )

        all_gene_exp = []
        for blk in self.blks:
            gene_exp, features = blk(gene_exp, features, coords, nearest_indices, attr_emb=attr_emb)
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)
        
        gene_exp, _ = to_dense_batch(gene_exp, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)
        gene_exp = gene_exp.squeeze(0)
        return gene_exp
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
