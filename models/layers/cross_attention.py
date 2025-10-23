from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

import pdb

"""

Contains the custom implementation of cross attention between pathways and histology and self attention between pathways 

"""

NUM_PATHWAYS = 1280

def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
        num_proteins = 0,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.num_proteins = num_proteins
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, eps = *x.shape, self.heads, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # Check if we have proteins (3-modality) or not (2-modality)
        if self.num_proteins > 0:
            # Three modality case: pathways, proteins, histology
            # Extract queries and keys for each modality
            q_pathways = q[:, :, :self.num_pathways, :]
            k_pathways = k[:, :, :self.num_pathways, :]
            v_pathways = v[:, :, :self.num_pathways, :]

            q_proteins = q[:, :, self.num_pathways:self.num_pathways+self.num_proteins, :]
            k_proteins = k[:, :, self.num_pathways:self.num_pathways+self.num_proteins, :]
            v_proteins = v[:, :, self.num_pathways:self.num_pathways+self.num_proteins, :]

            q_histology = q[:, :, self.num_pathways+self.num_proteins:, :]
            k_histology = k[:, :, self.num_pathways+self.num_proteins:, :]
            v_histology = v[:, :, self.num_pathways+self.num_proteins:, :]

            einops_eq = '... i d, ... j d -> ... i j'

            # Pathways attend to: pathways, proteins, histology
            attn_pp = einsum(einops_eq, q_pathways, k_pathways)
            attn_pprot = einsum(einops_eq, q_pathways, k_proteins)
            attn_phist = einsum(einops_eq, q_pathways, k_histology)
            attn_pathways_all = torch.cat([attn_pp, attn_pprot, attn_phist], dim=-1).softmax(dim=-1)

            # Proteins attend to: pathways, proteins, histology
            attn_protp = einsum(einops_eq, q_proteins, k_pathways)
            attn_protprot = einsum(einops_eq, q_proteins, k_proteins)
            attn_prothist = einsum(einops_eq, q_proteins, k_histology)
            attn_proteins_all = torch.cat([attn_protp, attn_protprot, attn_prothist], dim=-1).softmax(dim=-1)

            # Histology attends to: pathways, proteins (cross-modal only)
            attn_histp = einsum(einops_eq, q_histology, k_pathways)
            attn_histprot = einsum(einops_eq, q_histology, k_proteins)
            attn_histology_all = torch.cat([attn_histp, attn_histprot], dim=-1).softmax(dim=-1)

            # Aggregate values
            v_all = torch.cat([v_pathways, v_proteins, v_histology], dim=2)
            v_pathway_protein = torch.cat([v_pathways, v_proteins], dim=2)

            out_pathways = attn_pathways_all @ v_all
            out_proteins = attn_proteins_all @ v_all
            out_histology = attn_histology_all @ v_pathway_protein

            out = torch.cat([out_pathways, out_proteins, out_histology], dim=2)

            # For return attention (simplified for 3-modality)
            pre_softmax_cross_attn_histology = attn_histp
            attn_pathways = attn_pp
            cross_attn_pathways = attn_phist

        else:
            # Original 2-modality case: pathways and histology only
            q_pathways = q[:, :, :self.num_pathways, :]
            k_pathways = k[:, :, :self.num_pathways, :]

            q_histology = q[:, :, self.num_pathways:, :]
            k_histology = k[:, :, self.num_pathways:, :]

            # similarities
            einops_eq = '... i d, ... j d -> ... i j'
            cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
            attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
            cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)

            # softmax
            pre_softmax_cross_attn_histology = cross_attn_histology
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways =  attn_pathways_histology @ v
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

            out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways = 281,
        num_proteins = 0,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.num_proteins = num_proteins
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            num_proteins=num_proteins
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x
