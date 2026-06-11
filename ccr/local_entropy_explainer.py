from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .critic import EdgeMLPCritic, pairwise_ranking_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_MAS_ROOT = PROJECT_ROOT.parent
GDESIGNER_ROOT = LLM_MAS_ROOT / "GDesigner"
if GDESIGNER_ROOT.exists() and str(GDESIGNER_ROOT) not in sys.path:
    sys.path.insert(0, str(GDESIGNER_ROOT))

try:
    from GDesigner.llm.profile_embedding import get_sentence_embedding
    from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
except Exception:  # pragma: no cover - fallback keeps the trainer usable without GDesigner imports.
    get_sentence_embedding = None
    PromptSetRegistry = None


SCALAR_FEATURES = [
    "is_spatial",
    "is_temporal",
    "is_self_loop",
    "active_mask",
    "fixed_mask",
    "generator_logit_tanh",
    "edge_index_norm",
    "source_index_norm",
    "target_index_norm",
    "source_in_same_kind",
    "source_out_same_kind",
    "target_in_same_kind",
    "target_out_same_kind",
    "source_in_all",
    "source_out_all",
    "target_in_all",
    "target_out_all",
    "num_nodes_scaled",
    "active_same_kind_ratio",
    "active_all_ratio",
]


@dataclass
class LocalEntropyTrainingExample:
    dataset: str
    example_id: str
    edge_key: str
    edge_kind: str
    edge_index: int
    source: str
    target: str
    feature: torch.Tensor
    target_value: float
    weight: float
    metadata: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_embedding(text: Any, dim: int = 384) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in str(text or "").lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = vec.norm()
    return vec / norm if float(norm) > 0 else vec


def text_embedding(text: Any) -> torch.Tensor:
    if get_sentence_embedding is None:
        return _hash_embedding(text)
    try:
        value = get_sentence_embedding(str(text or ""))
        return torch.tensor(value, dtype=torch.float32).view(-1)
    except Exception:
        return _hash_embedding(text)


def role_description(graph_domain: str | None, role: str | None) -> str:
    role = role or ""
    if PromptSetRegistry is None or not graph_domain:
        return role
    try:
        prompt_set = PromptSetRegistry.get(graph_domain)
        return str(prompt_set.get_description(role) or role)
    except Exception:
        return role


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def edge_key(edge: dict[str, Any]) -> str:
    return f"{edge.get('kind')}:{edge.get('index')}:{edge.get('source')}->{edge.get('target')}"


def normalize_entropy(value: Any, samples: Any) -> float:
    n_samples = max(_safe_int(samples, 0), 2)
    denom = math.log(n_samples)
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, _safe_float(value) / denom))


def label_from_aggregate(aggregate: dict[str, Any], label_mode: str, cost_penalty: float = 0.0) -> float:
    samples = aggregate.get("samples", 0)
    source = normalize_entropy(aggregate.get("source_semantic_entropy"), samples)
    target = normalize_entropy(aggregate.get("target_semantic_entropy"), samples)
    pair = normalize_entropy(aggregate.get("pair_semantic_entropy"), samples)
    if label_mode == "source":
        label = source
    elif label_mode == "target":
        label = target
    elif label_mode == "pair":
        label = pair
    elif label_mode == "combined":
        label = 0.5 * pair + 0.25 * source + 0.25 * target
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    if cost_penalty > 0:
        usage = aggregate.get("usage") or {}
        cost = math.log1p(_safe_float(usage.get("total_tokens")))
        label -= cost_penalty * cost
    return max(0.0, min(1.0, label))


def load_architecture(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "graph_architecture.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_example_graphs(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset_dir / "example_graphs.jsonl")
    return {str(row.get("example_id")): row for row in rows}


def node_ids_from_architecture(architecture: dict[str, Any], edge: dict[str, Any] | None = None) -> list[str]:
    node_ids = [str(node_id) for node_id in architecture.get("node_ids") or []]
    if node_ids:
        return node_ids
    nodes = architecture.get("nodes") or []
    node_ids = [str(node.get("id")) for node in nodes if node.get("id") is not None]
    if node_ids:
        return node_ids
    if edge:
        return [str(edge.get("source")), str(edge.get("target"))]
    return []


def node_roles_from_architecture(architecture: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for node in architecture.get("nodes") or []:
        if node.get("id") is not None:
            roles[str(node["id"])] = str(node.get("role") or node.get("agent_name") or "")
    return roles


def node_feature_matrix_from_architecture(architecture: dict[str, Any], task: str) -> torch.Tensor:
    node_ids = node_ids_from_architecture(architecture)
    roles = node_roles_from_architecture(architecture)
    graph_domain = architecture.get("graph_domain") or architecture.get("dataset")
    task_vec = text_embedding(task)
    rows = []
    for node_id in node_ids:
        role_text = role_description(str(graph_domain), roles.get(node_id, ""))
        role_vec = text_embedding(role_text)
        if role_vec.numel() != task_vec.numel():
            role_vec = _hash_embedding(role_text, int(task_vec.numel()))
        rows.append(torch.cat([role_vec, task_vec], dim=0))
    if not rows:
        return torch.empty((0, int(task_vec.numel()) * 2), dtype=torch.float32)
    return torch.stack(rows).float()


def _potential_edges(architecture: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    key = "potential_spatial_edges" if kind == "spatial" else "potential_temporal_edges"
    edges = architecture.get(key) or []
    out = []
    for idx, edge in enumerate(edges):
        if isinstance(edge, dict):
            out.append({
                "kind": kind,
                "index": _safe_int(edge.get("index"), idx),
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
            })
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            out.append({"kind": kind, "index": idx, "source": str(edge[0]), "target": str(edge[1])})
    return out


def _edge_list_from_masks(
    architecture: dict[str, Any],
    graph_row: dict[str, Any],
    kind: str,
) -> list[dict[str, Any]]:
    active_key = "active_spatial_edges" if kind == "spatial" else "active_temporal_edges"
    active_edges = graph_row.get(active_key)
    if active_edges:
        return [
            {
                "kind": kind,
                "index": _safe_int(edge.get("index"), idx),
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
            }
            for idx, edge in enumerate(active_edges)
        ]

    potential = _potential_edges(architecture, kind)
    if kind == "spatial":
        masks = graph_row.get("spatial_masks") or architecture.get("fixed_spatial_masks") or []
    else:
        masks = graph_row.get("temporal_masks") or architecture.get("fixed_temporal_masks") or []
    out = []
    for edge in potential:
        idx = int(edge["index"])
        mask = _safe_float(masks[idx] if idx < len(masks) else 1.0)
        if mask > 0:
            out.append(edge)
    return out


def _degree_features(edges: list[dict[str, Any]], node_ids: list[str]) -> dict[str, tuple[float, float]]:
    in_degree = {node_id: 0.0 for node_id in node_ids}
    out_degree = {node_id: 0.0 for node_id in node_ids}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in out_degree:
            out_degree[source] += 1.0
        if target in in_degree:
            in_degree[target] += 1.0
    denom = max(float(len(edges)), 1.0)
    return {node_id: (in_degree[node_id] / denom, out_degree[node_id] / denom) for node_id in node_ids}


def _mask_value(values: Any, index: int, default: float = 1.0) -> float:
    if values is None or index < 0:
        return default
    if hasattr(values, "detach"):
        values = values.detach().cpu().view(-1).tolist()
    try:
        return _safe_float(values[index], default)
    except (IndexError, TypeError):
        return default


def edge_scalar_features(
    edge: dict[str, Any],
    node_ids: list[str],
    active_same_kind: list[dict[str, Any]],
    active_all: list[dict[str, Any]],
    potential_count: int,
    generator_logit: float,
    active_mask: float,
    fixed_mask: float,
) -> torch.Tensor:
    kind = str(edge.get("kind"))
    source = str(edge.get("source"))
    target = str(edge.get("target"))
    index = _safe_int(edge.get("index"), 0)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    src_idx = node_to_idx.get(source, 0)
    tgt_idx = node_to_idx.get(target, 0)
    same_kind_degree = _degree_features(active_same_kind, node_ids)
    all_degree = _degree_features(active_all, node_ids)
    src_same = same_kind_degree.get(source, (0.0, 0.0))
    tgt_same = same_kind_degree.get(target, (0.0, 0.0))
    src_all = all_degree.get(source, (0.0, 0.0))
    tgt_all = all_degree.get(target, (0.0, 0.0))
    n_nodes = max(len(node_ids), 1)
    n_possible = max(n_nodes * n_nodes, 1)
    scalars = [
        1.0 if kind == "spatial" else 0.0,
        1.0 if kind == "temporal" else 0.0,
        1.0 if source == target else 0.0,
        active_mask,
        fixed_mask,
        math.tanh(generator_logit),
        index / max(potential_count - 1, 1),
        src_idx / max(n_nodes - 1, 1),
        tgt_idx / max(n_nodes - 1, 1),
        src_same[0],
        src_same[1],
        tgt_same[0],
        tgt_same[1],
        src_all[0],
        src_all[1],
        tgt_all[0],
        tgt_all[1],
        n_nodes / 10.0,
        len(active_same_kind) / n_possible,
        len(active_all) / max(2 * n_possible, 1),
    ]
    return torch.tensor(scalars, dtype=torch.float32)


def cached_edge_feature(
    row: dict[str, Any],
    architecture: dict[str, Any],
    graph_row: dict[str, Any],
) -> torch.Tensor:
    edge = row["variant"]["edge"]
    kind = str(edge.get("kind"))
    node_ids = node_ids_from_architecture(architecture, edge)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_features = node_feature_matrix_from_architecture(architecture, str(row.get("task") or graph_row.get("task") or ""))
    if node_features.size(0) != len(node_ids):
        raise ValueError(f"Node feature mismatch for {row.get('dataset')} {row.get('example_id')}")

    source = str(edge.get("source"))
    target = str(edge.get("target"))
    source_feature = node_features[node_to_idx[source]] if source in node_to_idx else torch.zeros(node_features.size(1))
    target_feature = node_features[node_to_idx[target]] if target in node_to_idx else torch.zeros(node_features.size(1))
    active_spatial = _edge_list_from_masks(architecture, graph_row, "spatial")
    active_temporal = _edge_list_from_masks(architecture, graph_row, "temporal")
    active_same = active_spatial if kind == "spatial" else active_temporal
    potential = _potential_edges(architecture, kind)
    index = _safe_int(edge.get("index"), 0)

    if kind == "spatial":
        generator_logit = _mask_value(graph_row.get("spatial_logits"), index, 0.0)
        active_mask = _mask_value(graph_row.get("spatial_masks"), index, 1.0)
        fixed_mask = _mask_value(architecture.get("fixed_spatial_masks"), index, active_mask)
    else:
        generator_logit = _mask_value(graph_row.get("temporal_logits"), index, 0.0)
        active_mask = _mask_value(graph_row.get("temporal_masks"), index, 1.0)
        fixed_mask = _mask_value(architecture.get("fixed_temporal_masks"), index, active_mask)

    scalars = edge_scalar_features(
        edge=edge,
        node_ids=node_ids,
        active_same_kind=active_same,
        active_all=active_spatial + active_temporal,
        potential_count=max(len(potential), 1),
        generator_logit=generator_logit,
        active_mask=active_mask,
        fixed_mask=fixed_mask,
    )
    return torch.cat([source_feature, target_feature, scalars], dim=0).float()


def default_cache_roots() -> list[Path]:
    return [PROJECT_ROOT / "results"]


def _normalize_roots(cache_roots: Any | None) -> list[Path]:
    if cache_roots is None:
        env_value = os.environ.get("CCR_EXPLAINER_CACHE_ROOTS", "")
        if env_value.strip():
            return [Path(item).expanduser() for item in env_value.split(":") if item.strip()]
        return default_cache_roots()
    if isinstance(cache_roots, (str, Path)):
        return [Path(cache_roots).expanduser()]
    roots = []
    for item in cache_roots:
        if item is not None:
            roots.append(Path(item).expanduser())
    return roots or default_cache_roots()


def discover_local_entropy_files(
    cache_roots: Any | None = None,
    datasets: set[str] | list[str] | None = None,
    include_smoke: bool = False,
) -> list[Path]:
    wanted = set(datasets or [])
    files: set[Path] = set()
    for root in _normalize_roots(cache_roots):
        root = root.resolve()
        if root.is_file() and root.name == "local_entropy.jsonl":
            candidates = [root]
        elif (root / "local_entropy.jsonl").exists():
            candidates = [root / "local_entropy.jsonl"]
        elif root.exists():
            candidates = list(root.rglob("local_entropy.jsonl"))
        else:
            candidates = []
        for path in candidates:
            if not include_smoke and any("smoke" in part.lower() for part in path.parts):
                continue
            if wanted and path.parent.name not in wanted:
                try:
                    with path.open(encoding="utf-8") as f:
                        first = json.loads(next(line for line in f if line.strip()))
                    if first.get("dataset") not in wanted:
                        continue
                except Exception:
                    continue
            files.add(path.resolve())
    return sorted(files)


def load_local_entropy_training_examples(
    cache_roots: Any | None = None,
    datasets: set[str] | list[str] | None = None,
    label_mode: str = "combined",
    cost_penalty: float = 0.0,
    positive_weight: float = 4.0,
    include_smoke: bool = False,
) -> tuple[list[LocalEntropyTrainingExample], dict[str, Any]]:
    files = discover_local_entropy_files(cache_roots, datasets=datasets, include_smoke=include_smoke)
    examples: list[LocalEntropyTrainingExample] = []
    skipped = 0
    for path in files:
        dataset_dir = path.parent
        architecture = load_architecture(dataset_dir)
        example_graphs = load_example_graphs(dataset_dir)
        for row in read_jsonl(path):
            try:
                edge = row["variant"]["edge"]
                graph_row = example_graphs.get(str(row.get("example_id")), {})
                feature = cached_edge_feature(row, architecture, graph_row)
                target_value = label_from_aggregate(row.get("aggregate") or {}, label_mode, cost_penalty)
                weight = 1.0 + positive_weight * float(target_value > 1e-8) + positive_weight * target_value
                examples.append(LocalEntropyTrainingExample(
                    dataset=str(row.get("dataset") or dataset_dir.name),
                    example_id=str(row.get("example_id")),
                    edge_key=edge_key(edge),
                    edge_kind=str(edge.get("kind")),
                    edge_index=_safe_int(edge.get("index")),
                    source=str(edge.get("source")),
                    target=str(edge.get("target")),
                    feature=feature,
                    target_value=target_value,
                    weight=weight,
                    metadata={
                        "cache_file": str(path),
                        "variant": row.get("variant"),
                        "aggregate": row.get("aggregate"),
                    },
                ))
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    print(f"[local_entropy_explainer] skip {path}: {exc}", flush=True)

    summary = {
        "cache_files": [str(path) for path in files],
        "examples": len(examples),
        "skipped": skipped,
        "datasets": sorted({example.dataset for example in examples}),
        "label_mode": label_mode,
        "cost_penalty": cost_penalty,
    }
    return examples, summary


def runtime_edge_features(
    node_features: torch.Tensor,
    node_ids: list[str],
    spatial_edges: list[list[str]],
    temporal_edges: list[list[str]],
    kind: str,
    logits: Any = None,
    masks: Any = None,
    fixed_masks: Any = None,
) -> torch.Tensor:
    edges = spatial_edges if kind == "spatial" else temporal_edges
    if not edges:
        return torch.empty((0, node_features.size(1) * 2 + len(SCALAR_FEATURES)), dtype=node_features.dtype, device=node_features.device)

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    active_spatial = []
    for idx, edge in enumerate(spatial_edges):
        if _mask_value(masks if kind == "spatial" else None, idx, 1.0) > 0 or kind != "spatial":
            active_spatial.append({"kind": "spatial", "index": idx, "source": str(edge[0]), "target": str(edge[1])})
    active_temporal = []
    for idx, edge in enumerate(temporal_edges):
        active_temporal.append({"kind": "temporal", "index": idx, "source": str(edge[0]), "target": str(edge[1])})

    rows = []
    for idx, edge_pair in enumerate(edges):
        edge = {"kind": kind, "index": idx, "source": str(edge_pair[0]), "target": str(edge_pair[1])}
        source_idx = node_to_idx.get(edge["source"], 0)
        target_idx = node_to_idx.get(edge["target"], 0)
        active_same = active_spatial if kind == "spatial" else active_temporal
        scalar = edge_scalar_features(
            edge=edge,
            node_ids=node_ids,
            active_same_kind=active_same,
            active_all=active_spatial + active_temporal,
            potential_count=max(len(edges), 1),
            generator_logit=_mask_value(logits, idx, 0.0),
            active_mask=_mask_value(masks, idx, 1.0),
            fixed_mask=_mask_value(fixed_masks, idx, _mask_value(masks, idx, 1.0)),
        ).to(device=node_features.device, dtype=node_features.dtype)
        rows.append(torch.cat([node_features[source_idx], node_features[target_idx], scalar], dim=0))
    return torch.stack(rows)


class LocalEntropyEdgeExplainer(torch.nn.Module):
    """Amortized edge scorer trained from local semantic-entropy counterfactual cache."""

    def __init__(
        self,
        node_feature_dim: int,
        scalar_dim: int = len(SCALAR_FEATURES),
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.scalar_dim = int(scalar_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.scorer = EdgeMLPCritic(self.node_feature_dim * 2 + self.scalar_dim, hidden_dim, dropout)

    @property
    def input_dim(self) -> int:
        return self.node_feature_dim * 2 + self.scalar_dim

    def score_edge_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return torch.empty(0, dtype=features.dtype, device=features.device)
        return torch.sigmoid(self.scorer(features))

    def _node_scores(
        self,
        node_ids: list[str],
        spatial_edges: list[list[str]],
        temporal_edges: list[list[str]],
        spatial_scores: torch.Tensor,
        temporal_scores: torch.Tensor,
    ) -> torch.Tensor:
        values: list[list[torch.Tensor]] = [[] for _ in node_ids]
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        for edges, scores in ((spatial_edges, spatial_scores), (temporal_edges, temporal_scores)):
            for edge, score in zip(edges, scores):
                source = node_to_idx.get(str(edge[0]))
                target = node_to_idx.get(str(edge[1]))
                if source is not None:
                    values[source].append(score)
                if target is not None:
                    values[target].append(score)
        out = []
        device = spatial_scores.device if spatial_scores.numel() else temporal_scores.device
        dtype = spatial_scores.dtype if spatial_scores.numel() else temporal_scores.dtype
        for node_values in values:
            if node_values:
                out.append(torch.stack(node_values).mean())
            else:
                out.append(torch.tensor(0.5, dtype=dtype, device=device))
        return torch.stack(out) if out else torch.empty(0, dtype=dtype, device=device)

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor | None,
        node_ids: list[str],
        spatial_edges: list[list[str]],
        temporal_edges: list[list[str]],
        spatial_logits: Any = None,
        spatial_masks: Any = None,
        temporal_logits: Any = None,
        temporal_masks: Any = None,
        fixed_spatial_masks: Any = None,
        fixed_temporal_masks: Any = None,
    ) -> dict[str, torch.Tensor]:
        del adj
        node_features = features.float()
        spatial_feature_rows = runtime_edge_features(
            node_features,
            node_ids,
            spatial_edges,
            temporal_edges,
            "spatial",
            logits=spatial_logits,
            masks=spatial_masks,
            fixed_masks=fixed_spatial_masks,
        )
        temporal_feature_rows = runtime_edge_features(
            node_features,
            node_ids,
            spatial_edges,
            temporal_edges,
            "temporal",
            logits=temporal_logits,
            masks=temporal_masks,
            fixed_masks=fixed_temporal_masks,
        )
        spatial_scores = self.score_edge_features(spatial_feature_rows)
        temporal_scores = self.score_edge_features(temporal_feature_rows)
        node_scores = self._node_scores(node_ids, spatial_edges, temporal_edges, spatial_scores, temporal_scores)
        return {"node": node_scores, "spatial": spatial_scores, "temporal": temporal_scores}


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * weight).sum() / weight.sum().clamp_min(1e-8)


def _pearson(pred: list[float], target: list[float]) -> float:
    if len(pred) < 2:
        return 0.0
    mean_p = statistics.mean(pred)
    mean_t = statistics.mean(target)
    num = sum((p - mean_p) * (t - mean_t) for p, t in zip(pred, target))
    den_p = math.sqrt(sum((p - mean_p) ** 2 for p in pred))
    den_t = math.sqrt(sum((t - mean_t) ** 2 for t in target))
    return num / max(den_p * den_t, 1e-12)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _spearman(pred: list[float], target: list[float]) -> float:
    if len(pred) < 2:
        return 0.0
    return _pearson(_ranks(pred), _ranks(target))


def evaluate_edge_model(model: LocalEntropyEdgeExplainer, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> dict[str, Any]:
    if x.numel() == 0:
        return {"mse": 0.0, "weighted_mse": 0.0, "pearson": 0.0, "spearman": 0.0, "mean_pred": 0.0}
    model.eval()
    with torch.no_grad():
        pred = model.score_edge_features(x).detach().cpu()
    target = y.detach().cpu()
    weight = w.detach().cpu()
    mse = float(((pred - target) ** 2).mean().item())
    weighted_mse = float(_weighted_mse(pred, target, weight).item())
    pred_list = [float(v) for v in pred.tolist()]
    target_list = [float(v) for v in target.tolist()]
    return {
        "mse": mse,
        "weighted_mse": weighted_mse,
        "pearson": _pearson(pred_list, target_list),
        "spearman": _spearman(pred_list, target_list),
        "mean_pred": statistics.mean(pred_list) if pred_list else 0.0,
    }


def train_local_entropy_explainer_from_examples(
    examples: list[LocalEntropyTrainingExample],
    output_dir: Path,
    hidden_dim: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.15,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    ranking_weight: float = 0.25,
    seed: int = 888,
    checkpoint_name: str = "local_entropy_explainer.pt",
) -> tuple[LocalEntropyEdgeExplainer, dict[str, Any]]:
    if not examples:
        raise ValueError("No local-entropy training examples were loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    random.Random(seed).shuffle(examples)
    features = torch.stack([example.feature.float() for example in examples])
    targets = torch.tensor([example.target_value for example in examples], dtype=torch.float32)
    weights = torch.tensor([example.weight for example in examples], dtype=torch.float32)
    node_feature_dim = int((features.size(1) - len(SCALAR_FEATURES)) // 2)
    model = LocalEntropyEdgeExplainer(node_feature_dim, len(SCALAR_FEATURES), hidden_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_ratio)) if len(indices) >= 5 else 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] or indices
    log_path = output_dir / "local_entropy_explainer_training.jsonl"
    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_indices)
        epoch_losses = []
        for start in range(0, len(train_indices), max(batch_size, 1)):
            batch_idx = train_indices[start:start + max(batch_size, 1)]
            x = features[batch_idx]
            y = targets[batch_idx]
            w = weights[batch_idx]
            pred = model.score_edge_features(x)
            mse = _weighted_mse(pred, y, w)
            rank = pairwise_ranking_loss(pred, y)
            loss = mse + ranking_weight * rank
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = statistics.mean(epoch_losses) if epoch_losses else 0.0
        losses.append(mean_loss)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            train_metrics = evaluate_edge_model(model, features[train_indices], targets[train_indices], weights[train_indices])
            val_metrics = evaluate_edge_model(model, features[val_indices], targets[val_indices], weights[val_indices]) if val_indices else {}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "train": train_metrics,
                    "val": val_metrics,
                }, ensure_ascii=False) + "\n")

    labels = [example.target_value for example in examples]
    final_train = evaluate_edge_model(model, features[train_indices], targets[train_indices], weights[train_indices])
    final_val = evaluate_edge_model(model, features[val_indices], targets[val_indices], weights[val_indices]) if val_indices else {}
    checkpoint_path = output_dir / checkpoint_name
    training_info = {
        "model_type": "LocalEntropyEdgeExplainer",
        "examples": len(examples),
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "feature_dim": int(features.size(1)),
        "node_feature_dim": node_feature_dim,
        "scalar_features": SCALAR_FEATURES,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "ranking_weight": ranking_weight,
        "first_loss": losses[0] if losses else 0.0,
        "last_loss": losses[-1] if losses else 0.0,
        "label_stats": {
            "min": min(labels),
            "max": max(labels),
            "mean": statistics.mean(labels),
            "nonzero": sum(1 for value in labels if value > 1e-8),
        },
        "datasets": sorted({example.dataset for example in examples}),
        "train_metrics": final_train,
        "val_metrics": final_val,
        "training_log": str(log_path),
        "model_file": str(checkpoint_path),
    }
    torch.save({
        "state_dict": model.state_dict(),
        "training_info": training_info,
        "model_config": {
            "node_feature_dim": node_feature_dim,
            "scalar_dim": len(SCALAR_FEATURES),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
    }, checkpoint_path)
    write_jsonl(output_dir / "local_entropy_training_examples.jsonl", [
        {
            "dataset": example.dataset,
            "example_id": example.example_id,
            "edge_key": example.edge_key,
            "edge_kind": example.edge_kind,
            "edge_index": example.edge_index,
            "source": example.source,
            "target": example.target,
            "target_value": example.target_value,
            "weight": example.weight,
            "metadata": example.metadata,
        }
        for example in examples
    ])
    (output_dir / "local_entropy_explainer_summary.json").write_text(
        json.dumps(training_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, training_info


def train_local_entropy_explainer_from_cache(
    output_dir: Path,
    cache_roots: Any | None = None,
    datasets: set[str] | list[str] | None = None,
    label_mode: str = "combined",
    cost_penalty: float = 0.0,
    positive_weight: float = 4.0,
    include_smoke: bool = False,
    **train_kwargs: Any,
) -> tuple[LocalEntropyEdgeExplainer, dict[str, Any]]:
    examples, cache_info = load_local_entropy_training_examples(
        cache_roots=cache_roots,
        datasets=datasets,
        label_mode=label_mode,
        cost_penalty=cost_penalty,
        positive_weight=positive_weight,
        include_smoke=include_smoke,
    )
    model, info = train_local_entropy_explainer_from_examples(examples, output_dir=output_dir, **train_kwargs)
    info["cache"] = cache_info
    info["label_mode"] = label_mode
    info["cost_penalty"] = cost_penalty
    info["positive_weight"] = positive_weight
    (output_dir / "local_entropy_explainer_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, info


def train_local_entropy_explainer_for_graph(
    graph: Any,
    args: Any,
    run_dir: Path,
    dataset_name: str,
) -> tuple[LocalEntropyEdgeExplainer, dict[str, Any]]:
    del graph
    cache_roots = getattr(args, "explainer_cache_roots", None)
    use_auto = bool(getattr(args, "explainer_cache_auto", True))
    if not cache_roots and not use_auto and not os.environ.get("CCR_EXPLAINER_CACHE_ROOTS"):
        raise FileNotFoundError("No explainer cache roots configured.")
    output_dir = Path(run_dir)
    return train_local_entropy_explainer_from_cache(
        output_dir=output_dir,
        cache_roots=cache_roots,
        datasets=[dataset_name],
        label_mode=getattr(args, "explainer_label_mode", "combined"),
        cost_penalty=float(getattr(args, "explainer_cost_penalty", 0.0)),
        positive_weight=float(getattr(args, "explainer_positive_weight", 4.0)),
        include_smoke=bool(getattr(args, "explainer_include_smoke", False)),
        hidden_dim=int(getattr(args, "explainer_hidden_dim", 128)),
        epochs=int(getattr(args, "explainer_epochs", 200)),
        lr=float(getattr(args, "explainer_lr", 1e-3)),
        weight_decay=float(getattr(args, "explainer_weight_decay", 1e-4)),
        dropout=float(getattr(args, "explainer_dropout", 0.15)),
        batch_size=int(getattr(args, "explainer_batch_size", 64)),
        val_ratio=float(getattr(args, "explainer_val_ratio", 0.2)),
        ranking_weight=float(getattr(args, "explainer_ranking_weight", 0.25)),
        seed=int(getattr(args, "seed", 888)),
        checkpoint_name=f"{dataset_name}_local_entropy_explainer.pt",
    )


def load_local_entropy_explainer(path: Path) -> tuple[LocalEntropyEdgeExplainer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = LocalEntropyEdgeExplainer(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("training_info") or {}


def examples_to_dicts(examples: list[LocalEntropyTrainingExample]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        row = asdict(example)
        row.pop("feature", None)
        rows.append(row)
    return rows
