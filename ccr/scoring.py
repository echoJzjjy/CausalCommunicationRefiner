from __future__ import annotations

from dataclasses import asdict

from .schema import CommunicationEdge, CommunicationGraph, ComponentScore, RefinerWeights, RunTrace
from .similarity import cosine_bow_similarity


class CausalScoreCalculator:
    def __init__(
        self,
        weights: RefinerWeights | None = None,
        similarity_fn=cosine_bow_similarity,
        token_cost_scale: float = 1 / 1000,
    ) -> None:
        self.weights = weights or RefinerWeights()
        self.similarity_fn = similarity_fn
        self.token_cost_scale = token_cost_scale

    def quality_drop(self, base: RunTrace, counterfactual: RunTrace) -> float:
        if base.quality is None or counterfactual.quality is None:
            return 0.0
        return float(base.quality) - float(counterfactual.quality)

    def answer_change(self, base: RunTrace, counterfactual: RunTrace) -> float:
        return 1.0 - self.similarity_fn(base.final_answer, counterfactual.final_answer)

    def downstream_change(
        self,
        graph: CommunicationGraph,
        source_node: str,
        base: RunTrace,
        counterfactual: RunTrace,
    ) -> float:
        descendants = graph.descendants(source_node)
        if not descendants:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for node_id in descendants:
            before = base.node_outputs.get(node_id, "")
            after = counterfactual.node_outputs.get(node_id, "")
            change = 1.0 - self.similarity_fn(before, after)
            total += change
            weight_sum += 1.0
        return total / max(weight_sum, 1e-12)

    def usage(self, edge: CommunicationEdge, base: RunTrace) -> float:
        message = base.edge_messages.get(edge.key) or edge.message
        receiver_output = base.node_outputs.get(edge.target, "")
        if not message or not receiver_output:
            return 0.0
        return self.similarity_fn(message, receiver_output)

    def redundancy(self, graph: CommunicationGraph, edge: CommunicationEdge, base: RunTrace) -> float:
        message = base.edge_messages.get(edge.key) or edge.message
        if not message:
            return 0.0
        values = []
        for parent_edge in graph.parents(edge.target):
            if parent_edge.key == edge.key:
                continue
            other = base.edge_messages.get(parent_edge.key) or parent_edge.message
            if other:
                values.append(self.similarity_fn(message, other))
        return max(values) if values else 0.0

    def edge_cost(self, edge: CommunicationEdge, base: RunTrace) -> float:
        message = base.edge_messages.get(edge.key) or edge.message
        token_estimate = len(str(message).split())
        return token_estimate * self.token_cost_scale + 1.0

    def edge_score(
        self,
        graph: CommunicationGraph,
        edge: CommunicationEdge,
        base: RunTrace,
        counterfactual: RunTrace,
    ) -> ComponentScore:
        q_drop = self.quality_drop(base, counterfactual)
        a_change = self.answer_change(base, counterfactual)
        d_change = self.downstream_change(graph, edge.target, base, counterfactual)
        usage = self.usage(edge, base)
        redundancy = self.redundancy(graph, edge, base)
        cost = self.edge_cost(edge, base)
        score = (
            self.weights.alpha_quality * q_drop
            + self.weights.beta_answer * a_change
            + self.weights.gamma_downstream * d_change
            + self.weights.delta_usage * usage * (1.0 - redundancy)
            - self.weights.lambda_cost * cost
        )
        return ComponentScore(
            key=edge.key,
            kind="edge",
            score=score,
            quality_drop=q_drop,
            answer_change=a_change,
            downstream_change=d_change,
            usage=usage,
            redundancy=redundancy,
            cost=cost,
            metadata={"edge": asdict(edge)},
        )

    def node_cost(self, node_id: str, graph: CommunicationGraph) -> float:
        incident = [edge for edge in graph.edges if edge.source == node_id or edge.target == node_id]
        return 1.0 + len(incident) * self.token_cost_scale

    def node_score(
        self,
        graph: CommunicationGraph,
        node_id: str,
        base: RunTrace,
        counterfactual: RunTrace,
    ) -> ComponentScore:
        q_drop = self.quality_drop(base, counterfactual)
        a_change = self.answer_change(base, counterfactual)
        d_change = self.downstream_change(graph, node_id, base, counterfactual)
        cost = self.node_cost(node_id, graph)
        score = (
            self.weights.alpha_quality * q_drop
            + self.weights.beta_answer * a_change
            + self.weights.gamma_downstream * d_change
            - self.weights.lambda_cost * cost
        )
        return ComponentScore(
            key=f"node:{node_id}",
            kind="node",
            score=score,
            quality_drop=q_drop,
            answer_change=a_change,
            downstream_change=d_change,
            cost=cost,
            metadata={"node_id": node_id},
        )

