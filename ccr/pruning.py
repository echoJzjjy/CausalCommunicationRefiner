from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schema import CommunicationGraph, ComponentScore


@dataclass
class RefinedGraph:
    graph: CommunicationGraph
    removed_nodes: list[ComponentScore]
    removed_edges: list[ComponentScore]
    kept_nodes: list[str]
    kept_edges: list[str]


class TopKPruner:
    def __init__(
        self,
        edge_keep_ratio: float = 0.75,
        node_keep_ratio: float = 1.0,
        min_nodes: int = 2,
        order: Literal["nodes_then_edges", "edges_only", "nodes_only"] = "nodes_then_edges",
    ) -> None:
        self.edge_keep_ratio = edge_keep_ratio
        self.node_keep_ratio = node_keep_ratio
        self.min_nodes = min_nodes
        self.order = order

    def prune(
        self,
        graph: CommunicationGraph,
        node_scores: list[ComponentScore],
        edge_scores: list[ComponentScore],
    ) -> RefinedGraph:
        current = graph
        removed_nodes: list[ComponentScore] = []
        removed_edges: list[ComponentScore] = []

        if self.order in {"nodes_then_edges", "nodes_only"} and node_scores:
            node_by_id = {score.metadata.get("node_id", score.key.replace("node:", "")): score for score in node_scores}
            ranked_nodes = sorted(node_by_id.items(), key=lambda item: item[1].score, reverse=True)
            keep_count = max(self.min_nodes, int(round(len(ranked_nodes) * self.node_keep_ratio)))
            keep_nodes = {node_id for node_id, _score in ranked_nodes[:keep_count]}
            removed_nodes = [score for node_id, score in ranked_nodes[keep_count:] if node_id in current.nodes]
            for node_id in list(current.nodes):
                if node_id not in keep_nodes:
                    current = current.without_node(node_id)

        if self.order in {"nodes_then_edges", "edges_only"} and edge_scores:
            active_edge_keys = set(current.edge_keys())
            edge_by_key = {score.key: score for score in edge_scores if score.key in active_edge_keys}
            ranked_edges = sorted(edge_by_key.items(), key=lambda item: item[1].score, reverse=True)
            keep_count = int(round(len(ranked_edges) * self.edge_keep_ratio))
            keep_edges = {key for key, _score in ranked_edges[:keep_count]}
            removed_edges = [score for key, score in ranked_edges[keep_count:] if key in active_edge_keys]
            current = CommunicationGraph(
                nodes=dict(current.nodes),
                edges=[edge for edge in current.edges if edge.key in keep_edges],
                metadata=dict(current.metadata),
            )

        return RefinedGraph(
            graph=current,
            removed_nodes=removed_nodes,
            removed_edges=removed_edges,
            kept_nodes=list(current.nodes),
            kept_edges=current.edge_keys(),
        )
