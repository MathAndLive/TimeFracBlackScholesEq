import numpy as np
import pandas as pd
import time
from numba import njit, prange


def monte_carlo_numpy(S0: float,
                      K: float,
                      T: float,
                      r: float,
                      sigma: float,
                      alpha: float,
                      size: int = 5_000_000,
                      call: bool = True) -> tuple[float, float]:
    '''
    Calculates the price of a European option using Monte Carlo method with subordination
    for the time fractional Black-Scholes model.

    This model replaces the fixed time horizon T with a stochastic time E_T,
    derived from a stable subordinator.

    Args:
        S0 (float): Stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk free interest rate.
        sigma (float): Volatility.
        alpha (float): Order of the fractional derivative (alpha in (0, 1)).
        size (int): Number of simulations for Monte Carlo method.
        call (bool): True for a call option, False for a put option.

    Returns:
        mean (float): The estimated price of the European option.
        std (float): The standard deviation of the estimated price.
    '''

    if not (0 < alpha <= 1):
        raise ValueError(f"Invalid alpha: {alpha}. Alpha must be in (0, 1].")

    # Generating random variables from a stable distribution by Chambers-Mallows-Stuck method
    u = np.random.uniform(-np.pi / 2, np.pi / 2, size)
    w = np.random.exponential(1, size)
    term1 = (np.sin(alpha * (u + np.pi / 2))) / (np.cos(u) ** (1 / alpha))
    term2 = (np.cos(u - alpha * (u + np.pi / 2)) / w) ** ((1 - alpha) / alpha)
    s1 = term1 * term2

    # Inverse stable subordinator formula from Lemma 2.3 (https://arxiv.org/abs/2012.13904)
    e_t = (T / s1) ** alpha

    z_norm = np.random.randn(size)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * e_t +
                      sigma * np.sqrt(e_t) * z_norm)

    if call:
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    discounted_payoff = np.exp(-r * e_t) * payoff
    mean = np.mean(discounted_payoff)
    std = np.std(discounted_payoff, ddof=1) / np.sqrt(size)
    return mean, std


@njit(parallel=True, fastmath=True)
def monte_carlo(S0: float,
                K: float,
                T: float,
                r: float,
                sigma: float,
                alpha: float,
                size: int,
                call: bool) -> tuple[float, float]:
    '''
    The faster version of the Monte Carlo method using Numba.

    Calculates the price of a European option using Monte Carlo method with subordination
    for the time fractional Black-Scholes model.

    This model replaces the fixed time horizon T with a stochastic time E_T,
    derived from a stable subordinator.

    Args:
        S0 (float): Stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk free interest rate.
        sigma (float): Volatility.
        alpha (float): Order of the fractional derivative (alpha in (0, 1)).
        size (int): Number of simulations for Monte Carlo method.
        call (bool): True for a call option, False for a put option.

    Returns:
        mean (float): The estimated price of the European option.
        std (float): The standard deviation of the estimated price.
    '''

    if not (0 < alpha <= 1):
        raise ValueError("Invalid alpha. Must be in (0,1].")

    sum_payoff = 0.0
    sum_payoff_sq = 0.0

    for i in prange(size):
        # Generating random variables from a stable distribution by Chambers-Mallows-Stuck method
        u = np.random.uniform(-np.pi/2, np.pi/2)
        w = np.random.exponential(1.0)
        term1 = np.sin(alpha * (u + np.pi/2)) / (np.cos(u) ** (1/alpha))
        term2 = (np.cos(u - alpha * (u + np.pi/2)) /
                 w) ** ((1 - alpha) / alpha)
        s1 = term1 * term2

        # Inverse stable subordinator formula from Lemma 2.3 (https://arxiv.org/abs/2012.13904)
        e_t = (T / s1) ** alpha

        z = np.random.randn()
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * e_t +
                          sigma * np.sqrt(e_t) * z)

        if call:
            payoff = S_T - K if S_T > K else 0.0
        else:
            payoff = K - S_T if S_T < K else 0.0

        discounted = np.exp(-r * e_t) * payoff

        sum_payoff += discounted
        sum_payoff_sq += discounted * discounted

    mean = sum_payoff / size
    var_mean = (sum_payoff_sq / size - mean * mean) / size
    std_error = np.sqrt(var_mean)
    return mean, std_error


# if __name__ == '__main__':
    # S0, K, T, r, sigma, alpha = 100.0, 110.0, 1.0, 0.05, 0.2, 0.8
    # N = 1_000_000
    # repeats = 10

    # # Warm up Numba compilation
    # monte_carlo(S0, K, T, r, sigma, alpha, 10_000, True)

    # # Timing loops
    # times_numpy = []
    # times_numba = []

    # for _ in range(repeats):
    #     start = time.perf_counter()
    #     monte_carlo(S0, K, T, r, sigma, alpha, N, True)
    #     times_numpy.append(time.perf_counter() - start)

    #     start = time.perf_counter()
    #     monte_carlo(S0, K, T, r, sigma, alpha, N, True)
    #     times_numba.append(time.perf_counter() - start)

    # # Prepare results
    # avg_numpy = sum(times_numpy) / repeats
    # avg_numba = sum(times_numba) / repeats
    # speedup = avg_numpy / avg_numba

    # df = pd.DataFrame({
    #     "Function": ["NumPy", "Numba"],
    #     "Avg Time [s]": [avg_numpy, avg_numba],
    #     "Speedup": [1.0, speedup]
    # })

    # print(df.to_string(index=False))

    # # Warm up Numba compilation
    # monte_carlo(S0, K, T, r, sigma, alpha, 10_000, True)

    # # Compute results
    # results = []
    # for call in [True, False]:
    #     mean_np, se_np = monte_carlo(S0, K, T, r, sigma, alpha, N, call)
    #     mean_nb, se_nb = monte_carlo(S0, K, T, r, sigma, alpha, N, call)
    #     opt = 'Call' if call else 'Put'
    #     results.append({'Option': opt, 'Function': 'NumPy',
    #                    'Mean': mean_np, 'StdError': se_np})
    #     results.append({'Option': opt, 'Function': 'Numba',
    #                    'Mean': mean_nb, 'StdError': se_nb})

    # df = pd.DataFrame(results)
    # print(df.to_string(index=False))

    # Parameter grids
    # S0_list = [90.0, 100.0, 110.0]
    # K_list = [90.0, 100.0, 110.0]
    # T_list = [0.5, 1.0, 2.0]
    # r_list = [0.03, 0.05]
    # sigma_list = [0.15, 0.2]
    # alpha_list = [0.8, 1.0]

    # # Simulation settings
    # N = 1_000_000

    # # Warm up Numba
    # monte_carlo(100.0, 100.0, 1.0, 0.05, 0.2, 0.8, 10_000, True)

    # # Collect results
    # records = []
    # for S0 in S0_list:
    #     for K in K_list:
    #         for T in T_list:
    #             for r in r_list:
    #                 for sigma in sigma_list:
    #                     for alpha in alpha_list:
    #                         for call in [True, False]:
    #                             mean_np, se_np = monte_carlo(
    #                                 S0, K, T, r, sigma, alpha, N, call)
    #                             mean_nb, se_nb = monte_carlo(
    #                                 S0, K, T, r, sigma, alpha, N, call)
    #                             opt = 'Call' if call else 'Put'
    #                             records.append({
    #                                 'S0': S0,
    #                                 'K': K,
    #                                 'T': T,
    #                                 'r': r,
    #                                 'sigma': sigma,
    #                                 'alpha': alpha,
    #                                 'Option': opt,
    #                                 'Mean_NumPy': mean_np,
    #                                 'SE_NumPy': se_np,
    #                                 'Mean_Numba': mean_nb,
    #                                 'SE_Numba': se_nb,
    #                                 'Diff_Mean': mean_nb - mean_np
    #                             })

    # df = pd.DataFrame(records)
    # print(df.to_string(index=False))

    # if __name__ == '__main__':
    #     S0 = 100
    #     K = 110
    #     T = 1.0
    #     r = 0.05
    #     sigma = 0.2
    #     alpha = 0.8
    #     N = 5_000_000

    #     price_call, _ = monte_carlo(S0, K, T, r, sigma, alpha, N, call=True)
    #     price_put, _ = monte_carlo(S0, K, T, r, sigma, alpha, N, call=False)
    #     print(
    #         f'(CALL) Monte-Carlo method (alpha = {alpha}) price = {price_call:.4f}')
    #     print(
    #         f'(PUT) Monte-Carlo method (alpha = {alpha}) price = {price_put:.4f}\n')

    #     alpha_classic = 1.0
    #     price_classic_call, _ = monte_carlo(
    #         S0, K, T, r, sigma, alpha_classic, N, call=True)
    #     price_classic_put, _ = monte_carlo(
    #         S0, K, T, r, sigma, alpha_classic, N, call=False)
    #     print(
    #         f'(CALL) Monte-Carlo method (alpha = {alpha_classic}) price = {price_classic_call:.4f}')
    #     print(
    #         f'(PUT) Monte-Carlo method (alpha = {alpha_classic}) price = {price_classic_put:.4f}\n')

    #     '''
    #     Output

    #     S0 = 100
    #     K = 110
    #     T = 1.0
    #     r = 0.05
    #     sigma = 0.2
    #     alpha = 0.8
    #     N = 5_000_000

    #     (CALL) Monte-Carlo method (alpha = 0.8) price = 6.3281
    #     (PUT) Monte-Carlo method (alpha = 0.8) price = 10.6220

    #     (CALL) Monte-Carlo method (alpha = 1.0) price = 6.0435
    #     (PUT) Monte-Carlo method (alpha = 1.0) price = 10.6790
    #     '''
