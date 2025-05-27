import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Tuple


def generate_random_digraph(n: int, edge_prob: float = 0.3) -> nx.DiGraph:
    """
    Generuje losowy digraf z co najmniej jedną wychodzącą krawędzią z każdego wierzchołka.
    """
    if n < 2:
        raise ValueError("Graf musi mieć co najmniej 2 wierzchołki")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for node in G.nodes:
        possible_targets = [i for i in range(n) if i != node]
        targets = set()
        while not targets:
            targets = {i for i in possible_targets if random.random() < edge_prob}
        for target in targets:
            G.add_edge(node, target)
    return G


def pagerank_random_walk(graph: nx.DiGraph, d: float = 0.15, steps: int = 1000000) -> Dict[int, float]:
    """
    Implementacja PageRank metodą błądzenia przypadkowego z teleportacją
    """
    visit_count = {node: 0 for node in graph.nodes}
    current = random.choice(list(graph.nodes))  # początek z losowego wierzchołka

    for _ in range(steps):
        if random.random() < d:  # teleportacja do losowego wierzchołka
            current = random.choice(list(graph.nodes))
        else:  # przejście do losowego sąiedniego wierzchołka
            neighbors = list(graph.successors(current))
            current = random.choice(neighbors) if neighbors else random.choice(list(graph.nodes))
        visit_count[current] += 1

    # PR jako liczba odwiedzin wierzchhołka/liczba kroków
    total_visits = sum(visit_count.values())
    return {node: count / total_visits for node, count in visit_count.items()}


def pagerank_power_iteration(graph: nx.DiGraph, d: float = 0.15, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict[int, float], int]:
    """
    Implementacja PageRank metodą metodą potęgową
    """
    n = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, dtype=float)  # macierz sąsiedztwa grafu
    out_degree = A.sum(axis=1)  # wektor stopni wyjściowych

    # Konstruowanie macierzy stochastycznej P
    P = np.zeros((n, n))
    for i in range(n):
        if out_degree[i] > 0:  # sprawdzenie stopnia wychodzącego wierzchołka
            for j in range(n):
                P[i, j] = (1 - d) * A[i, j] / out_degree[i] + d / n
        else:
            for j in range(n):
                P[i, j] = 1 / n  # przypadek wierzchołka bez wychodzących krawędzi

    p = np.ones(n) / n  # dla t=0 wektor p = (1/n, 1/n, ...)
    num_iter = 0
    for it in range(1, max_iter + 1):
        p_new = p @ P  # p(t+1) = p(t) * P
        if np.linalg.norm(p_new - p, ord=1) < tol:  # kryterium zbieżności
            num_iter = it
            break
        p = p_new

    return {i: val for i, val in enumerate(p)}, num_iter


def print_pagerank_results(results: Dict[int, float], header: str) -> None:
    """
    Printuje posortowane wyniki PageRank
    """
    print(header)
    sorted_ranks = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (node, score) in enumerate(sorted_ranks, start=1):
        print(f"{rank} ({node}) ==> PageRank = {score:.6f}")


def draw_graph(graph: nx.DiGraph) -> None:
    """
    Rysuje digraf funkcją nx.draw_networkx
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, arrows=True, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Losowy digraf")
    plt.show()


if __name__ == "__main__":
    try:
        # parametry
        n = 12  # liczba wierzchołków
        d = 0.15
        steps = 1_000_000

        # generowanie grafu
        graph = generate_random_digraph(n, edge_prob=0.3)
        draw_graph(graph)

        # Algorytm (a) - Random walk z teleportacją
        pr_scores_walk = pagerank_random_walk(graph, d=d, steps=steps)
        print_pagerank_results(pr_scores_walk, f"(a) Błądzenie przypadkowe z teleportacją po N = {steps} krokach:")

        # Algorytm (b) - Metoda potęgowa
        pr_scores_power, iterations = pagerank_power_iteration(graph, d=d)
        print_pagerank_results(pr_scores_power, f"\n(b) Metoda potęgowa po {iterations} iteracjach:")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
