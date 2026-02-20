"""Stage 9b: Typed hypergraph embedding via Relational GCN.

Embeds each thread's typed hypergraph into a fixed-dimensional vector.
Uses R-GCN (one weight matrix per edge type) with self-supervised
contrastive training.

Architecture:
    node_features  →  R-GCN (2 layers)  →  mean_pool  →  projection  →  embedding

Node features:  learned embeddings for (node_type, subtype_hash)
Edge types:     iatc, mention, discourse, scope, surface, categorical

Training:       graph contrastive learning — augment each graph via random
                edge/node dropout, train InfoNCE loss to pull augmentations
                of the same graph together and push different graphs apart.

    >>> from futon6.graph_embed import embed_hypergraphs, train
    >>> model, embeddings, stats = train(hypergraphs, dim=128, epochs=50)
    >>> embeddings.shape  # (n_threads, 128)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ---------------------------------------------------------------------------
# Edge type registry — must match hypergraph.py EDGE_TYPES
# ---------------------------------------------------------------------------

EDGE_TYPES = ["iatc", "mention", "discourse", "scope", "surface", "categorical"]
EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}
N_EDGE_TYPES = len(EDGE_TYPES)

NODE_TYPES = ["post", "term", "expression", "scope"]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
N_NODE_TYPES = len(NODE_TYPES)

# Subtype vocabulary size (hashed into this many buckets)
SUBTYPE_BUCKETS = 256


# ---------------------------------------------------------------------------
# Graph data structure (PyTorch tensors, no torch_geometric dependency)
# ---------------------------------------------------------------------------

class GraphBatch:
    """A batch of typed graphs in COO format.

    Attributes:
        x:       (N, feat_dim)  node feature indices
        edge_index: dict[int, (2, E_r)]  per-relation-type edge indices
        batch:   (N,)  graph membership for each node
        n_graphs: int
    """
    def __init__(self, x, edge_index, batch, n_graphs):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.n_graphs = n_graphs

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = {
            k: v.to(device) for k, v in self.edge_index.items()
        }
        self.batch = self.batch.to(device)
        return self


# ---------------------------------------------------------------------------
# Hypergraph → tensor conversion
# ---------------------------------------------------------------------------

def _subtype_hash(subtype: str) -> int:
    """Hash a subtype string into a bucket index."""
    return int(hashlib.md5(subtype.encode()).hexdigest(), 16) % SUBTYPE_BUCKETS


def hypergraph_to_tensors(hg: dict) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Convert a hypergraph dict to node features and per-type edge indices.

    Returns:
        x:  (N, 2) int tensor — [node_type_idx, subtype_hash]
        edge_index: dict mapping edge_type_idx → (2, E) long tensor
    """
    nodes = hg["nodes"]
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Node features: [type_idx, subtype_hash]
    x = torch.zeros(n, 2, dtype=torch.long)
    for i, node in enumerate(nodes):
        type_idx = NODE_TYPE_TO_IDX.get(node["type"], 0)
        sub_hash = _subtype_hash(node.get("subtype", ""))
        x[i, 0] = type_idx
        x[i, 1] = sub_hash

    # Edge indices per relation type
    edge_index: dict[int, list[list[int]]] = {i: [[], []] for i in range(N_EDGE_TYPES)}

    for edge in hg["edges"]:
        etype = EDGE_TYPE_TO_IDX.get(edge["type"])
        if etype is None:
            continue
        ends = edge["ends"]
        if len(ends) < 2:
            # Unary edge (discourse, categorical) — self-loop
            src_idx = node_id_to_idx.get(ends[0])
            if src_idx is not None:
                edge_index[etype][0].append(src_idx)
                edge_index[etype][1].append(src_idx)
        else:
            # Binary edge — add both directions
            src_idx = node_id_to_idx.get(ends[0])
            tgt_idx = node_id_to_idx.get(ends[1])
            if src_idx is not None and tgt_idx is not None:
                edge_index[etype][0].append(src_idx)
                edge_index[etype][1].append(tgt_idx)
                # Reverse direction for message passing
                edge_index[etype][0].append(tgt_idx)
                edge_index[etype][1].append(src_idx)

    # Convert to tensors, drop empty relations
    ei_tensors = {}
    for etype_idx, (src, tgt) in edge_index.items():
        if src:
            ei_tensors[etype_idx] = torch.tensor([src, tgt], dtype=torch.long)

    return x, ei_tensors


def collate_graphs(graphs: list[tuple[torch.Tensor, dict[int, torch.Tensor]]]) -> GraphBatch:
    """Collate multiple graphs into a single batched graph."""
    xs = []
    edge_indices: dict[int, list[torch.Tensor]] = {i: [] for i in range(N_EDGE_TYPES)}
    batches = []
    offset = 0

    for g_idx, (x, ei) in enumerate(graphs):
        n = x.size(0)
        xs.append(x)
        batches.append(torch.full((n,), g_idx, dtype=torch.long))
        for etype_idx, edges in ei.items():
            edge_indices[etype_idx].append(edges + offset)
        offset += n

    batch_x = torch.cat(xs, dim=0)
    batch_batch = torch.cat(batches, dim=0)
    batch_ei = {}
    for etype_idx, edges_list in edge_indices.items():
        if edges_list:
            batch_ei[etype_idx] = torch.cat(edges_list, dim=1)

    return GraphBatch(batch_x, batch_ei, batch_batch, len(graphs))


# ---------------------------------------------------------------------------
# Graph augmentation for contrastive learning
# ---------------------------------------------------------------------------

def augment_graph(x: torch.Tensor, edge_index: dict[int, torch.Tensor],
                  node_drop: float = 0.1, edge_drop: float = 0.2,
                  ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Augment a graph by randomly dropping nodes and edges.

    Returns new (x, edge_index) with consistent re-indexing.
    """
    n = x.size(0)

    # Node dropout: keep a random subset
    if node_drop > 0 and n > 2:
        keep_mask = torch.rand(n) > node_drop
        # Always keep at least 2 nodes
        if keep_mask.sum() < 2:
            keep_mask[:2] = True
    else:
        keep_mask = torch.ones(n, dtype=torch.bool)

    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    new_x = x[keep_indices]

    # Build old→new index mapping
    old_to_new = torch.full((n,), -1, dtype=torch.long)
    old_to_new[keep_indices] = torch.arange(len(keep_indices))

    # Edge dropout + re-index
    new_ei = {}
    for etype_idx, edges in edge_index.items():
        if edges.size(1) == 0:
            continue
        # Re-index
        src_new = old_to_new[edges[0]]
        tgt_new = old_to_new[edges[1]]
        valid = (src_new >= 0) & (tgt_new >= 0)

        # Random edge drop
        if edge_drop > 0:
            edge_keep = torch.rand(valid.sum()) > edge_drop
            valid_edges = torch.stack([src_new[valid], tgt_new[valid]])
            if edge_keep.any():
                new_ei[etype_idx] = valid_edges[:, edge_keep]
        else:
            if valid.any():
                new_ei[etype_idx] = torch.stack([src_new[valid], tgt_new[valid]])

    return new_x, new_ei


# ---------------------------------------------------------------------------
# R-GCN model
# ---------------------------------------------------------------------------

class RGCNLayer(nn.Module):
    """One layer of Relational Graph Convolutional Network.

    For each edge type r, computes:
        h_i^{(l+1)} = σ( Σ_r  Σ_{j∈N_r(i)}  (1/|N_r(i)|) W_r h_j^{(l)} + W_0 h_i^{(l)} )

    Uses basis decomposition to reduce parameters when n_relations is large.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int,
                 n_bases: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations

        if n_bases is not None and n_bases < n_relations:
            # Basis decomposition: W_r = Σ_b a_{rb} V_b
            self.n_bases = n_bases
            self.bases = nn.Parameter(torch.Tensor(n_bases, in_dim, out_dim))
            self.coeffs = nn.Parameter(torch.Tensor(n_relations, n_bases))
            self.weight = None
        else:
            # Full weight matrices per relation
            self.n_bases = None
            self.weight = nn.Parameter(torch.Tensor(n_relations, in_dim, out_dim))
            self.bases = None
            self.coeffs = None

        self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight, gain=gain)
        if self.bases is not None:
            nn.init.xavier_uniform_(self.bases, gain=gain)
            nn.init.xavier_uniform_(self.coeffs, gain=gain)
        nn.init.xavier_uniform_(self.self_weight, gain=gain)
        nn.init.zeros_(self.bias)

    def _get_relation_weight(self, r: int) -> torch.Tensor:
        if self.weight is not None:
            return self.weight[r]
        # Basis decomposition
        return torch.sum(self.coeffs[r].unsqueeze(-1).unsqueeze(-1) * self.bases, dim=0)

    def forward(self, x: torch.Tensor,
                edge_index: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: dict[relation_idx → (2, E_r)]
        returns: (N, out_dim)
        """
        n = x.size(0)
        out = torch.zeros(n, self.out_dim, device=x.device)

        for r_idx, edges in edge_index.items():
            if r_idx >= self.n_relations or edges.size(1) == 0:
                continue
            src, tgt = edges[0], edges[1]
            W_r = self._get_relation_weight(r_idx)
            msg = x[src] @ W_r  # (E, out_dim)

            # Aggregate with mean normalization
            deg = torch.zeros(n, 1, device=x.device)
            deg.scatter_add_(0, tgt.unsqueeze(1).expand_as(msg[:, :1]),
                             torch.ones(msg.size(0), 1, device=x.device))
            deg = deg.clamp(min=1)

            out.scatter_add_(0, tgt.unsqueeze(1).expand_as(msg), msg)

        # Normalize by total incoming degree
        total_deg = torch.zeros(n, 1, device=x.device)
        for edges in edge_index.values():
            if edges.size(1) > 0:
                total_deg.scatter_add_(
                    0, edges[1].unsqueeze(1),
                    torch.ones(edges.size(1), 1, device=x.device))
        total_deg = total_deg.clamp(min=1)
        out = out / total_deg

        # Self-connection + bias
        out = out + x @ self.self_weight + self.bias
        out = self.dropout(F.relu(out))
        return out


class ThreadGNN(nn.Module):
    """R-GCN encoder + mean-pool readout + projection head.

    Produces a fixed-dim embedding vector per graph.
    """

    def __init__(self, hidden_dim: int = 128, embed_dim: int = 128,
                 n_layers: int = 2, n_relations: int = N_EDGE_TYPES,
                 n_bases: int | None = None, dropout: float = 0.1):
        super().__init__()

        # Node feature embeddings
        self.type_embed = nn.Embedding(N_NODE_TYPES, hidden_dim // 2)
        self.subtype_embed = nn.Embedding(SUBTYPE_BUCKETS, hidden_dim // 2)

        # R-GCN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(RGCNLayer(
                hidden_dim, hidden_dim, n_relations,
                n_bases=n_bases, dropout=dropout))

        # Projection head (for contrastive learning)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Convert node feature indices to dense embeddings."""
        type_emb = self.type_embed(x[:, 0])
        sub_emb = self.subtype_embed(x[:, 1])
        return torch.cat([type_emb, sub_emb], dim=-1)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        """Forward pass: node features → R-GCN → mean pool → projection.

        Returns (n_graphs, embed_dim) tensor.
        """
        h = self.encode_nodes(batch.x)

        for layer in self.layers:
            h = layer(h, batch.edge_index)

        # Mean pool per graph
        out = self._mean_pool(h, batch.batch, batch.n_graphs)

        # Project
        return self.proj(out)

    def embed(self, batch: GraphBatch) -> torch.Tensor:
        """Produce L2-normalized embeddings (for inference, no projection head)."""
        h = self.encode_nodes(batch.x)
        for layer in self.layers:
            h = layer(h, batch.edge_index)
        out = self._mean_pool(h, batch.batch, batch.n_graphs)
        out = self.proj(out)
        return F.normalize(out, dim=-1)

    @staticmethod
    def _mean_pool(h: torch.Tensor, batch: torch.Tensor,
                   n_graphs: int) -> torch.Tensor:
        """Mean pool node embeddings per graph."""
        out = torch.zeros(n_graphs, h.size(1), device=h.device)
        count = torch.zeros(n_graphs, 1, device=h.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
        count.scatter_add_(0, batch.unsqueeze(1),
                           torch.ones(h.size(0), 1, device=h.device))
        return out / count.clamp(min=1)


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent / InfoNCE)
# ---------------------------------------------------------------------------

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent contrastive loss between two augmented views.

    z1, z2: (B, D) L2-normalized embeddings of the same B graphs
            under two different augmentations.

    Positive pairs: (z1[i], z2[i])
    Negative pairs: (z1[i], z2[j]) for i != j, and (z1[i], z1[j])
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)

    # Similarity matrix
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.T / temperature      # (2B, 2B)

    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# DataLoader for contrastive training
# ---------------------------------------------------------------------------

class ContrastiveGraphDataset(Dataset):
    """Wraps pre-converted graph tensors for contrastive DataLoader.

    Each __getitem__ returns two augmented views of the same graph,
    enabling CPU-side augmentation in worker processes while the GPU trains.
    """

    def __init__(self, graph_tensors, node_drop=0.1, edge_drop=0.2):
        self.graph_tensors = graph_tensors
        self.node_drop = node_drop
        self.edge_drop = edge_drop

    def __len__(self):
        return len(self.graph_tensors)

    def __getitem__(self, idx):
        x, ei = self.graph_tensors[idx]
        v1 = augment_graph(x, ei, self.node_drop, self.edge_drop)
        v2 = augment_graph(x, ei, self.node_drop, self.edge_drop)
        return v1, v2


def contrastive_collate_fn(batch):
    """Collate (view1, view2) pairs into two GraphBatches."""
    view1 = [item[0] for item in batch]
    view2 = [item[1] for item in batch]
    return collate_graphs(view1), collate_graphs(view2)


def _split_train_val_indices(
    n: int,
    val_frac: float = 0.1,
    min_val: int = 32,
    max_val: int = 2048,
    seed: int = 1337,
) -> tuple[list[int], list[int]]:
    """Create a deterministic train/val split for contrastive evaluation."""
    if n < 4:
        return list(range(n)), []

    if n < 200:
        val_n = max(2, int(n * val_frac))
    else:
        val_n = max(min_val, int(n * val_frac))
    val_n = min(val_n, max_val, n - 2)
    if val_n < 2:
        return list(range(n)), []

    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    val_idx = sorted(idx[:val_n])
    train_idx = sorted(idx[val_n:])
    if len(train_idx) < 2:
        return list(range(n)), []
    return train_idx, val_idx


def _embed_graph_list(
    model: nn.Module,
    graphs: list[tuple[torch.Tensor, dict[int, torch.Tensor]]],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Embed a list of tensorized graphs and return (N, D) torch tensor on device."""
    all_out = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[start:start + batch_size]).to(device)
            emb = model.embed(batch)
            all_out.append(emb)
    return torch.cat(all_out, dim=0) if all_out else torch.zeros(0, 1, device=device)


def _paired_retrieval_metrics(
    model: nn.Module,
    graph_tensors: list[tuple[torch.Tensor, dict[int, torch.Tensor]]],
    indices: list[int],
    batch_size: int,
    device: str,
    node_drop: float,
    edge_drop: float,
) -> dict[str, float]:
    """Compute paired-view retrieval quality on held-out graphs.

    For each graph i, build two augmentations A_i and B_i. Then for each A_i,
    retrieve nearest B_j under cosine and check whether j=i.
    """
    if len(indices) < 2:
        return {
            "acc_at_1": 0.0,
            "acc_at_5": 0.0,
            "mrr": 0.0,
            "n_eval": len(indices),
        }

    view1 = []
    view2 = []
    for idx in indices:
        x, ei = graph_tensors[idx]
        v1 = augment_graph(x, ei, node_drop=node_drop, edge_drop=edge_drop)
        v2 = augment_graph(x, ei, node_drop=node_drop, edge_drop=edge_drop)
        view1.append(v1)
        view2.append(v2)

    z1 = _embed_graph_list(model, view1, batch_size=batch_size, device=device)
    z2 = _embed_graph_list(model, view2, batch_size=batch_size, device=device)
    if z1.size(0) == 0 or z2.size(0) == 0:
        return {"acc_at_1": 0.0, "acc_at_5": 0.0, "mrr": 0.0, "n_eval": 0}

    sim = z1 @ z2.T  # normalized embeddings => cosine similarity
    order = torch.argsort(sim, dim=1, descending=True)
    target = torch.arange(order.size(0), device=order.device)

    top1 = order[:, :1]
    k5 = min(5, order.size(1))
    top5 = order[:, :k5]

    acc1 = (top1.squeeze(1) == target).float().mean().item()
    acc5 = (top5 == target.unsqueeze(1)).any(dim=1).float().mean().item()

    # Reciprocal rank of the true pair.
    match = (order == target.unsqueeze(1)).nonzero(as_tuple=False)
    rr = torch.zeros(order.size(0), device=order.device)
    rr[match[:, 0]] = 1.0 / (match[:, 1].float() + 1.0)
    mrr = rr.mean().item()

    return {
        "acc_at_1": float(acc1),
        "acc_at_5": float(acc5),
        "mrr": float(mrr),
        "n_eval": int(order.size(0)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(hypergraphs: list[dict], dim: int = 128, hidden_dim: int = 128,
          n_layers: int = 2, epochs: int = 50, batch_size: int = 64,
          lr: float = 1e-3, node_drop: float = 0.1, edge_drop: float = 0.2,
          device: str | None = None, verbose: bool = True,
          num_workers: int = 4,
          tensor_cache_path: str | None = None,
          val_frac: float = 0.1,
          val_max_graphs: int = 2048,
          eval_every: int = 1,
          ) -> tuple[ThreadGNN, np.ndarray, dict[str, Any]]:
    """Self-supervised contrastive training of the thread GNN.

    Parameters
    ----------
    hypergraphs : list of dicts from Stage 9a
    dim : embedding dimension
    epochs : training epochs
    device : 'cuda', 'cpu', or None (auto-detect)
    num_workers : DataLoader workers for CPU-side augmentation (0 = inline)
    tensor_cache_path : if set, save/load tensorized graphs here (.pt)

    Returns
    -------
    (model, embeddings, training_stats) where embeddings is (n_threads, dim)
    numpy array and training_stats includes loss and paired-view retrieval
    metrics (Acc@1/Acc@5) on a held-out validation split.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if eval_every <= 0:
        eval_every = 1

    # Try loading pre-tensorized cache
    if tensor_cache_path and Path(tensor_cache_path).exists():
        if verbose:
            print(f"       Loading tensor cache from {tensor_cache_path}...")
        graph_tensors, _cached_ids = load_tensor_cache(tensor_cache_path)
        if verbose:
            print(f"       {len(graph_tensors)} graphs loaded from cache")
    else:
        # Convert all hypergraphs to tensor form
        if verbose:
            print(f"       Converting {len(hypergraphs)} hypergraphs to tensors...")
        graph_tensors = []
        for hg in hypergraphs:
            try:
                x, ei = hypergraph_to_tensors(hg)
                if x.size(0) >= 2:  # need at least 2 nodes
                    graph_tensors.append((x, ei))
            except Exception:
                continue

        # Save tensor cache for future runs
        if tensor_cache_path and graph_tensors:
            thread_ids = [hg.get("thread_id", i) for i, hg in enumerate(hypergraphs)]
            if verbose:
                print(f"       Saving tensor cache to {tensor_cache_path}...")
            save_tensor_cache(graph_tensors, thread_ids, tensor_cache_path)

    if len(graph_tensors) < 2:
        raise ValueError(f"Need at least 2 valid graphs, got {len(graph_tensors)}")

    all_graph_tensors = graph_tensors
    train_idx, val_idx = _split_train_val_indices(
        len(all_graph_tensors),
        val_frac=val_frac,
        max_val=val_max_graphs,
    )
    train_graphs = [all_graph_tensors[i] for i in train_idx]

    if verbose:
        print(f"       {len(all_graph_tensors)} valid graphs, training on {device}...")
        if val_idx:
            print(f"       Validation split: {len(train_idx)} train / {len(val_idx)} val"
                  f" (paired retrieval Acc@1/Acc@5)")
        else:
            print("       Validation split: disabled (too few graphs)")

    model = ThreadGNN(
        hidden_dim=hidden_dim, embed_dim=dim,
        n_layers=n_layers, n_relations=N_EDGE_TYPES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop — use DataLoader when num_workers > 0
    n_train = len(train_graphs)
    use_dataloader = num_workers > 0 and n_train >= batch_size
    use_cuda = device != "cpu" and torch.cuda.is_available()

    if use_dataloader:
        dataset = ContrastiveGraphDataset(train_graphs, node_drop, edge_drop)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=contrastive_collate_fn,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=(n_train > batch_size),  # drop tiny trailing batch
        )

    loss_curve = []
    val_curve = []
    best_val_acc1 = -1.0
    best_val_epoch = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        if use_dataloader:
            for batch1, batch2 in loader:
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                z1 = model(batch1)
                z2 = model(batch2)

                loss = info_nce_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
        else:
            # Fallback: manual batching (for small datasets or debugging)
            indices = list(range(n_train))
            random.shuffle(indices)

            for start in range(0, n_train, batch_size):
                batch_idx = indices[start:start + batch_size]
                if len(batch_idx) < 2:
                    continue

                view1 = []
                view2 = []
                for idx in batch_idx:
                    x, ei = train_graphs[idx]
                    x1, ei1 = augment_graph(x, ei, node_drop, edge_drop)
                    x2, ei2 = augment_graph(x, ei, node_drop, edge_drop)
                    view1.append((x1, ei1))
                    view2.append((x2, ei2))

                b1 = collate_graphs(view1).to(device)
                b2 = collate_graphs(view2).to(device)

                z1 = model(b1)
                z2 = model(b2)

                loss = info_nce_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_curve.append(float(avg_loss))

        val_metrics = None
        eval_now = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == epochs)
        if eval_now and val_idx:
            model.eval()
            val_metrics = _paired_retrieval_metrics(
                model=model,
                graph_tensors=all_graph_tensors,
                indices=val_idx,
                batch_size=batch_size,
                device=device,
                node_drop=node_drop,
                edge_drop=edge_drop,
            )
            val_metrics["epoch"] = int(epoch + 1)
            val_curve.append(val_metrics)
            if val_metrics["acc_at_1"] > best_val_acc1:
                best_val_acc1 = val_metrics["acc_at_1"]
                best_val_epoch = int(epoch + 1)

        log_now = ((epoch + 1) % max(1, epochs // 10) == 0) or ((epoch + 1) == epochs)
        if verbose and log_now:
            msg = f"       Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}"
            if val_metrics is not None:
                msg += ("  val_acc@1={:.3f} val_acc@5={:.3f} val_mrr={:.3f} n={}".format(
                    val_metrics["acc_at_1"],
                    val_metrics["acc_at_5"],
                    val_metrics["mrr"],
                    val_metrics["n_eval"],
                ))
            print(msg)

    # Final embedding pass
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(all_graph_tensors), batch_size):
            batch_graphs = all_graph_tensors[start:start + batch_size]
            batch = collate_graphs(batch_graphs).to(device)
            emb = model.embed(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    final_val = val_curve[-1] if val_curve else None
    training_stats = {
        "n_graphs_total": len(all_graph_tensors),
        "n_graphs_train": len(train_idx),
        "n_graphs_val": len(val_idx),
        "val_fraction": float(val_frac),
        "eval_every": int(eval_every),
        "loss_final": float(loss_curve[-1]) if loss_curve else None,
        "loss_best": float(min(loss_curve)) if loss_curve else None,
        "loss_curve": loss_curve,
        "best_val_acc1": float(best_val_acc1) if best_val_acc1 >= 0 else None,
        "best_val_epoch": best_val_epoch,
        "val_acc1_final": (float(final_val["acc_at_1"]) if final_val else None),
        "val_acc5_final": (float(final_val["acc_at_5"]) if final_val else None),
        "val_mrr_final": (float(final_val["mrr"]) if final_val else None),
        "val_curve": val_curve,
    }

    if verbose:
        print(f"       Embeddings: {embeddings.shape}")
        if final_val is not None:
            print("       Validation summary: "
                  f"best_acc@1={training_stats['best_val_acc1']:.3f} "
                  f"(epoch {training_stats['best_val_epoch']}), "
                  f"final_acc@1={training_stats['val_acc1_final']:.3f}, "
                  f"final_acc@5={training_stats['val_acc5_final']:.3f}")

    return model, embeddings, training_stats


# ---------------------------------------------------------------------------
# Inference: embed new hypergraphs with a trained model
# ---------------------------------------------------------------------------

def embed_hypergraphs(model: ThreadGNN, hypergraphs: list[dict],
                      batch_size: int = 128,
                      device: str | None = None) -> np.ndarray:
    """Embed hypergraphs using a trained model.

    Returns (n, dim) numpy array of L2-normalized embeddings.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    graph_tensors = []
    valid_indices = []
    for i, hg in enumerate(hypergraphs):
        try:
            x, ei = hypergraph_to_tensors(hg)
            if x.size(0) >= 1:
                graph_tensors.append((x, ei))
                valid_indices.append(i)
        except Exception:
            continue

    if not graph_tensors:
        dim = model.proj[-1].out_features
        return np.zeros((len(hypergraphs), dim), dtype=np.float32)

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(graph_tensors), batch_size):
            batch_graphs = graph_tensors[start:start + batch_size]
            batch = collate_graphs(batch_graphs).to(device)
            emb = model.embed(batch)
            all_embeddings.append(emb.cpu().numpy())

    valid_embs = np.concatenate(all_embeddings, axis=0)

    # Fill in for invalid graphs (zero vector)
    dim = valid_embs.shape[1]
    result = np.zeros((len(hypergraphs), dim), dtype=np.float32)
    for i, vi in enumerate(valid_indices):
        result[vi] = valid_embs[i]

    return result


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_tensor_cache(graph_tensors: list[tuple[torch.Tensor, dict[int, torch.Tensor]]],
                      thread_ids: list,
                      path: str) -> None:
    """Save pre-tensorized hypergraphs to disk for fast reload.

    Avoids re-parsing multi-GB JSON and re-running hypergraph_to_tensors()
    on every Stage 9b invocation.
    """
    # Pack into a serializable format: list of (x, {int: tensor}) pairs
    packed = []
    for x, ei in graph_tensors:
        packed.append({
            "x": x,
            "ei": {str(k): v for k, v in ei.items()},
        })
    torch.save({"graphs": packed, "thread_ids": thread_ids}, path)


def load_tensor_cache(path: str) -> tuple[list[tuple[torch.Tensor, dict[int, torch.Tensor]]], list]:
    """Load pre-tensorized hypergraphs from a .pt cache.

    Returns (graph_tensors, thread_ids).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    graph_tensors = []
    for item in data["graphs"]:
        x = item["x"]
        ei = {int(k): v for k, v in item["ei"].items()}
        graph_tensors.append((x, ei))
    return graph_tensors, data["thread_ids"]


def save_model(model: ThreadGNN, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "hidden_dim": model.type_embed.embedding_dim * 2,
            "embed_dim": model.proj[-1].out_features,
            "n_layers": len(model.layers),
            "n_relations": model.layers[0].n_relations,
        },
    }, path)


def load_model(path: str, device: str = "cpu") -> ThreadGNN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ThreadGNN(
        hidden_dim=config["hidden_dim"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_relations=config["n_relations"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
