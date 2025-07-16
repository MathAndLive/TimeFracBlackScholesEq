import numpy as np


def monte_carlo(S0: float,
                K: float,
                T: float,
                r: float,
                sigma: float,
                alpha: float,
                size: int = 5_000_000,
                call: bool = True) -> float:
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
        float: The estimated price of the European option.
    '''

    if not (0 < alpha <= 1):
        raise ValueError(f"Invalid alpha: {alpha}. Alpha must be in (0, 1].")

    # Generating random variables from a stable distributio by Chambers-Mallows-Stuck method
    u = np.random.uniform(-np.pi / 2, np.pi / 2, size)
    w = np.random.exponential(1, size)
    term1 = (np.sin(alpha * (u + np.pi / 2))) / (np.cos(u) ** (1 / alpha))
    term2 = (np.cos(u - alpha * (u + np.pi / 2)) / w) ** ((1 - alpha) / alpha)
    s1 = term1 * term2

    # Inverse stable subordinator formula from Lemma 2.3 (https://arxiv.org/abs/2012.13904)
    e_t = (T / s1) ** alpha

    z_norm = np.random.randn(size)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * e_t + sigma * np.sqrt(e_t) * z_norm)

    if call:
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    discounted_payoff = np.exp(-r * e_t) * payoff
    return np.mean(discounted_payoff)


if __name__ == '__main__':
    S0 = 100
    K = 110
    T = 1.0
    r = 0.05
    sigma = 0.2
    alpha = 0.8
    N = 5_000_000

    price_call = monte_carlo(S0, K, T, r, sigma, alpha, N, call=True)
    price_put = monte_carlo(S0, K, T, r, sigma, alpha, N, call=False)
    print(f'(CALL) Monte-Carlo method (alpha = {alpha}) price = {price_call:.4f}')
    print(f'(PUT) Monte-Carlo method (alpha = {alpha}) price = {price_put:.4f}\n')

    alpha_classic = 1.0
    price_classic_call = monte_carlo(S0, K, T, r, sigma, alpha_classic, N, call=True)
    price_classic_put = monte_carlo(S0, K, T, r, sigma, alpha_classic, N, call=False)
    print(f'(CALL) Monte-Carlo method (alpha = {alpha_classic}) price = {price_classic_call:.4f}')
    print(f'(PUT) Monte-Carlo method (alpha = {alpha_classic}) price = {price_classic_put:.4f}\n')

    '''
    Output
    
    S0 = 100
    K = 110
    T = 1.0
    r = 0.05
    sigma = 0.2
    alpha = 0.8
    N = 5_000_000
    
    (CALL) Monte-Carlo method (alpha = 0.8) price = 6.3281
    (PUT) Monte-Carlo method (alpha = 0.8) price = 10.6220
    
    (CALL) Monte-Carlo method (alpha = 1.0) price = 6.0435
    (PUT) Monte-Carlo method (alpha = 1.0) price = 10.6790
    '''
