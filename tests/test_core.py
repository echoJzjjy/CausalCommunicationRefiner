from ccr import AgentNode, CommunicationEdge, CommunicationGraph, RunTrace
from ccr.pruning import TopKPruner
from ccr.scoring import CausalScoreCalculator


def toy_graph() -> CommunicationGraph:
    nodes = {
        "a": AgentNode("a", "Solver"),
        "b": AgentNode("b", "Verifier"),
        "c": AgentNode("c", "Writer"),
    }
    edges = [
        CommunicationEdge("a", "b", message="answer is 4"),
        CommunicationEdge("b", "c", message="verified answer is 4"),
        CommunicationEdge("a", "c", message="answer is 4"),
    ]
    return CommunicationGraph(nodes=nodes, edges=edges)


def test_edge_score_quality_drop_positive():
    graph = toy_graph()
    edge = graph.edges[0]
    base = RunTrace(
        task="2+2",
        final_answer="4",
        quality=1.0,
        node_outputs={"a": "4", "b": "answer is 4", "c": "final answer 4"},
        edge_messages={edge.key: "answer is 4"},
    )
    counterfactual = RunTrace(
        task="2+2",
        final_answer="5",
        quality=0.0,
        node_outputs={"a": "4", "b": "guess 5", "c": "final answer 5"},
    )
    score = CausalScoreCalculator().edge_score(graph, edge, base, counterfactual)
    assert score.score > 0
    assert score.quality_drop == 1.0


def test_topk_pruner_keeps_high_score_edges():
    graph = toy_graph()
    edge_scores = []
    for idx, edge in enumerate(graph.edges):
        edge_scores.append(
            CausalScoreCalculator().edge_score(
                graph,
                edge,
                RunTrace(task="", final_answer="x", quality=1.0),
                RunTrace(task="", final_answer="x", quality=float(idx)),
            )
        )
    refined = TopKPruner(edge_keep_ratio=1 / 3, node_keep_ratio=1.0).prune(graph, [], edge_scores)
    assert len(refined.kept_edges) == 1

