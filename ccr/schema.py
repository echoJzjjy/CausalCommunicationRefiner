from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ComponentKind = Literal["node", "edge"]


@dataclass(frozen=True)
class AgentNode:
    id: str
    role: str
    description: str = ""


@dataclass(frozen=True)
class CommunicationEdge:
    source: str
    target: str
    kind: str = "spatial"
    message: str = ""
    generator_score: float | None = None

    @property
    def key(self) -> str:
        return f"edge:{self.kind}:{self.source}->{self.target}"


@dataclass
class CommunicationGraph:
    nodes: dict[str, AgentNode]
    edges: list[CommunicationEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def edge_keys(self) -> list[str]:
        return [edge.key for edge in self.edges]

    def node_keys(self) -> list[str]:
        return [f"node:{node_id}" for node_id in self.nodes]

    def parents(self, node_id: str) -> list[CommunicationEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def descendants(self, node_id: str) -> set[str]:
        adjacency: dict[str, list[str]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        seen: set[str] = set()
        queue = list(adjacency.get(node_id, []))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            queue.extend(adjacency.get(current, []))
        return seen

    def without_edge(self, key: str) -> "CommunicationGraph":
        return CommunicationGraph(
            nodes=dict(self.nodes),
            edges=[edge for edge in self.edges if edge.key != key],
            metadata=dict(self.metadata),
        )

    def without_node(self, node_id: str) -> "CommunicationGraph":
        return CommunicationGraph(
            nodes={key: node for key, node in self.nodes.items() if key != node_id},
            edges=[edge for edge in self.edges if edge.source != node_id and edge.target != node_id],
            metadata=dict(self.metadata),
        )


@dataclass
class RunTrace:
    task: str
    final_answer: str
    quality: float | None = None
    node_outputs: dict[str, str] = field(default_factory=dict)
    edge_messages: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinerWeights:
    alpha_quality: float = 1.0
    beta_answer: float = 0.2
    gamma_downstream: float = 0.2
    delta_usage: float = 0.2
    lambda_cost: float = 0.05


@dataclass
class ComponentScore:
    key: str
    kind: ComponentKind
    score: float
    quality_drop: float = 0.0
    answer_change: float = 0.0
    downstream_change: float = 0.0
    usage: float = 0.0
    redundancy: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

