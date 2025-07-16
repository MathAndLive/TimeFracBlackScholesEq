import numpy as np
import matplotlib.pyplot as plt


def sample_s_distribution(alpha: float = 0.5, gamma: float = 1, num_generations: int = 1) -> np.ndarray:
    """
    Генерация случайных величин из a-стабильного распределения методом Chambers Mallows Stuck (CMS)
    Параметры:
        alpha           - индекс устойчивости (0 < alpha < 1)
        gamma           - масштабный параметр, обычно равен dtau**(1/alpha)
        num_generations - число независимых случайных скачков для генерации

    Возвращает:
        np.ndarray      - массив из num_generations значений случайных величин
    """
    pi = np.pi
    U = np.random.uniform(-pi / 2, pi / 2, size=num_generations)
    W = np.random.exponential(1, size=num_generations)

    U_0 = -1 / 2 * pi * (1 - abs(1 - alpha)) / alpha
    term1 = np.sin(alpha * (U - U_0)) / np.power(np.cos(U), 1 / alpha)
    term2 = np.power(np.cos(U - alpha * (U - U_0)) / W, (1 - alpha) / alpha)

    values = term1 * term2 * gamma

    return values


def compute_first_crossing(alpha: float,
                           dtau: float,
                           T: float,
                           num_generations: int = 1,
                           return_path: bool = False) -> np.ndarray:
    """
    Симулирует субординатор до первого пересечения уровня T
    Параметры:
        alpha            - параметр распределения
        dtau             - малость разбиения
        T                - время экспирации
        return_path=True - вернёт всю траекторию также (arr, grid)
    Возвращает:
        tau              - момент первого пересечения (i * dtau)
        (opt.) arr, grid - массив значений процесса и сетку времени
    """
    gamma = dtau ** (1 / alpha)
    arr = []
    crossed = False
    i = 0

    while not crossed:
        dS = sample_s_distribution(alpha, gamma)[0]
        if i == 0:
            arr.append(dS)
        else:
            arr.append(arr[-1] + dS)
        i += 1
        if arr[-1] >= T:
            tau = i * dtau
            crossed = True

    if return_path:
        arr = np.array(arr)
        grid = np.arange(1, arr.size + 1) * dtau
        return tau, arr, grid
    else:
        return tau


def plot_subordinator(arr: np.ndarray,
                      grid: np.ndarray,
                      T: float,
                      tau: float,
                      alpha: float):
    """
    Строит график траектории субординатора с пометкой уровня T и момента tau
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, arr, lw=1)
    ax.axhline(T, color='red', ls='--', label=f'Level T={T}')
    ax.axvline(tau, color='green', ls='--', lw=0.5, label=f'τ={tau:.6f}')
    ax.scatter([tau], [arr[-1]], s=10, color='purple', label=f'τ={tau:.6f}')
    ax.set_xlabel("t")
    ax.set_ylabel("S(t)")
    ax.set_title(f"Субординатор, α = {alpha}")
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    for alpha in np.linspace(0.01, 1.0, 100):
        print(f'alpha = {alpha:.4f} mean = {sample_s_distribution(alpha, 1, 100000).mean():.4f}')