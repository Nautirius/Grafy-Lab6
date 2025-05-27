import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple


def read_points(filepath: str) -> list:
    """
    Wczytuje punkty z pliku tekstowego
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return [tuple(map(float, line.strip().split())) for line in lines]
    except Exception as e:
        raise ValueError(f"Błąd podczas wczytywania punktów: {e}")


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Obliczanie odległości euklidesowej między dwoma punktami
    """
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def total_distance(cycle: List[int], points: List[Tuple[float, float]]) -> float:
    """
    Oblicza całkowitą długość cyklu
    """
    return sum(distance(points[cycle[i]], points[cycle[(i + 1) % len(cycle)]]) for i in range(len(cycle)))


def two_opt_swap(route: List[int], i: int, k: int) -> List[int]:
    """
    Wykonuje operację 2-opt między indeksami i oraz k
    """
    return route[:i] + route[i:k+1][::-1] + route[k+1:]  # odcinek od i do k zostaje odwrócony, a pozostała trasa niezmieniona


def draw_cycle(points: List[Tuple[float, float]], cycle: List[int], title: str) -> None:
    """
    Rysuje cykl na wykresie
    """
    x = [points[i][0] for i in cycle + [cycle[0]]]
    y = [points[i][1] for i in cycle + [cycle[0]]]
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, 'ro-')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


def select_two_non_adjacent_edges(n: int) -> Tuple[int, int]:
    """
    Losuje dwie różne, niesąsiednie krawędzie z cyklu
    """
    while True:
        i, j = sorted(random.sample(range(n), 2))
        if abs(i - j) <= 1 or abs(i - j) >= n - 1:  # nie są sąsiednie (i nie są na końcach)
            continue
        return i, j


def simulated_annealing(points: List[Tuple[float, float]], max_iter: int = 1000) -> Tuple[List[int], float]:
    """
    Symulowane wyżarzanie z 2-opt
    """
    n = len(points)
    current_cycle = list(range(n))  # początkowo cykl według kolejności wierzchołków
    current_length = total_distance(current_cycle, points)

    best_cycle = current_cycle[:]
    best_length = current_length

    for i in range(100, 1, -1):
        T = 0.001 * i * i  # zmiana temperatury
        for _ in range(max_iter):
            a, b = select_two_non_adjacent_edges(n)  # losowanie 2 krawędzi

            new_cycle = two_opt_swap(current_cycle, a, b)  # zamiana krawędzi 2-opt
            new_length = total_distance(new_cycle, points)

            delta = new_length - current_length

            # akceptacja lepszego rozwiązania lub gorszego, z prawdopodobieństwem zależnym od T
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cycle = new_cycle
                current_length = new_length

                if new_length < best_length:  # aktualizacja najlepszego rozwiązania
                    best_cycle = new_cycle
                    best_length = new_length

    return best_cycle, best_length


if __name__ == "__main__":
    try:
        points = read_points("data/tsp_data.txt")

        best_overall_cycle = []
        best_overall_length = float('inf')

        for trial in range(10):
            cycle, length = simulated_annealing(points, max_iter=2000)
            print(f"Uruchomienie {trial + 1}: długość = {length:.3f}")
            if length < best_overall_length:
                best_overall_length = length
                best_overall_cycle = cycle

        draw_cycle(points, best_overall_cycle, f"Najlepszy cykl. Długość: {best_overall_length:.3f}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
