# Causal Communication Refiner: Method Pipeline

## 1. Method Positioning

Causal Communication Refiner is a generator-agnostic post-processing module for LLM-based multi-agent communication graphs.

It does not replace the original topology generator. Instead, it takes the graph produced by an existing generator and refines it by estimating which agents and communication edges are causally useful.

The overall pipeline is:

$$
\text{Topology Generator}
\rightarrow
\text{Candidate Graph}
\rightarrow
\text{Causal Communication Refiner}
\rightarrow
\text{Refined Graph}
$$

For a task $x$ and agent set $\mathcal{A}$, a generator first produces:

$$
G_0 = \mathcal{G}(x,\mathcal{A}).
$$

The proposed refiner then produces:

$$
G^\star = \mathcal{R}_\theta(x,\mathcal{A},G_0),
$$

where $G^\star$ is a smaller and more effective subgraph of $G_0$.

## 2. Problem Definition

Let:

$$
\mathcal{A}=\{a_1,a_2,\dots,a_N\}
$$

be the agents in the multi-agent system. The candidate communication graph is:

$$
G_0=(V,E_0),
$$

where $V$ is the agent node set and $E_0$ is the directed communication edge set.

An edge:

$$
e_{ij}=(a_i\rightarrow a_j)
$$

means that the output or intermediate message from agent $a_i$ is passed to agent $a_j$.

The goal is to find a refined graph:

$$
G^\star=(V^\star,E^\star),
$$

where:

$$
V^\star \subseteq V,\quad E^\star \subseteq E_0.
$$

The desired property is:

$$
\text{Quality}(x,G^\star)\approx \text{Quality}(x,G_0),
$$

while reducing communication cost:

$$
\text{Cost}(x,G^\star)<\text{Cost}(x,G_0).
$$

## 3. Stage 1: Candidate Graph Generation

The method starts from an existing topology generator:

$$
G_0=\mathcal{G}(x,\mathcal{A}).
$$

The generator can be any graph construction method, such as:

- Complete graph
- Random graph
- Chain graph
- Star graph
- Layered graph
- G-Designer
- AgentPrune
- AgentDropout
- Any task-specific topology generator

The refiner treats this generator as a black box. It only requires the generated graph $G_0$.

## 4. Stage 2: Causal Label Construction

The refiner estimates the contribution of each graph component by counterfactual intervention.

First, run the original candidate graph:

$$
Y_0,T_0=\text{MAS}(x,G_0),
$$

where $Y_0$ is the final answer and $T_0$ is the communication trace.

For each edge $e_{ij}\in E_0$, construct a counterfactual graph:

$$
G_0^{-e_{ij}}=G_0\setminus \{e_{ij}\}.
$$

Then run:

$$
Y_{-e_{ij}},T_{-e_{ij}}=\text{MAS}(x,G_0^{-e_{ij}}).
$$

The edge contribution is defined as the performance change caused by removing the edge:

$$
s_{ij}=Q(Y_0,T_0,x)-Q(Y_{-e_{ij}},T_{-e_{ij}},x).
$$

Here $Q(\cdot)$ is a verifier that measures answer or reasoning quality. It can be implemented as an automatic evaluator, rule-based task metric, test execution, or an LLM judge, depending on the benchmark.

Similarly, for each node $a_i\in V$, construct:

$$
G_0^{-a_i}=G_0\setminus \{a_i\},
$$

and compute:

$$
s_i=Q(Y_0,T_0,x)-Q(Y_{-a_i},T_{-a_i},x).
$$

These node and edge scores form causal supervision labels.

## 5. Stage 3: Explainer Training

The explainer learns to predict the causal usefulness of nodes and edges from the task and graph structure.

For a task $x$, graph $G_0$, node $a_i$, and edge $e_{ij}$, the explainer predicts:

$$
\hat{s}_i=f_\theta(x,G_0,a_i),
$$

$$
\hat{s}_{ij}=f_\theta(x,G_0,e_{ij}).
$$

The training objective is to fit the counterfactual labels:

$$
\mathcal{L}_{\text{score}}
=
\sum_i
\left(\hat{s}_i-s_i\right)^2
+
\sum_{(i,j)}
\left(\hat{s}_{ij}-s_{ij}\right)^2.
$$

To encourage compact communication graphs, add sparsity regularization:

$$
\mathcal{L}
=
\mathcal{L}_{\text{score}}
+
\lambda_s \mathcal{L}_{\text{sparse}}
+
\lambda_e \mathcal{L}_{\text{entropy}}.
$$

The explainer is trained on calibration examples only.

## 6. Stage 4: Test-Time Graph Refinement

At test time, the original generator first produces a candidate graph:

$$
G_0^{test}=\mathcal{G}(x^{test},\mathcal{A}).
$$

The trained explainer predicts node and edge usefulness:

$$
\hat{s}_i=f_\theta(x^{test},G_0^{test},a_i),
$$

$$
\hat{s}_{ij}=f_\theta(x^{test},G_0^{test},e_{ij}).
$$

Then low-score nodes and edges are pruned:

$$
V^\star
=
\text{TopK}_V(\hat{s}_i),
$$

$$
E^\star
=
\text{TopK}_E(\hat{s}_{ij}).
$$

The final refined graph is:

$$
G^\star=(V^\star,E^\star).
$$

The multi-agent system is then run on the refined graph:

$$
Y^\star=\text{MAS}(x^{test},G^\star).
$$

## 7. Train-Test Protocol

The protocol is:

1. Use calibration data to run counterfactual interventions.
2. Use the intervention results to construct causal labels.
3. Train the explainer to predict node and edge contribution.
4. Freeze the explainer.
5. On test data, use the frozen explainer to prune the generator-produced graph.
6. Run the multi-agent system on the refined graph.
7. Use test labels only for final evaluation.

The key separation is:

$$
\text{Calibration labels are used to train the explainer.}
$$

$$
\text{Test labels are not used to choose the graph.}
$$

## 8. Summary

The full method can be summarized as:

$$
G_0=\mathcal{G}(x,\mathcal{A})
$$

$$
s=\text{CounterfactualAudit}(x,G_0,Q)
$$

$$
f_\theta=\text{TrainExplainer}(x,G_0,s)
$$

$$
G^\star=\text{Prune}(G_0,f_\theta(x,G_0))
$$

$$
Y^\star=\text{MAS}(x,G^\star)
$$

Thus, the method converts expensive counterfactual graph auditing into a learned post-hoc explainer that can refine arbitrary topology generators at test time.
