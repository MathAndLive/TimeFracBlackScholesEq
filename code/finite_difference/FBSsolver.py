import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from tabulate import tabulate
from time import time
from numpy.typing import NDArray
from typing import Union


def adequacy_checker(S_max: np.ndarray = np.array([1]),
                     S0: float = 1,
                     K: float = 1,
                     T_max: float = 1,
                     T: float = 1,
                     r: float = 1,
                     sigma: float = 1,
                     alpha: float = 0.5,
                     M: int = 100,
                     N: int = 100) -> None:
    '''
    Checks whether all the parameters are in their intended domains, raises
    ValueError if they do not.

    Args:
      S_max (float)   - maximal base asset price, should be positive
      S0 (np.ndarray) - array of price values of the base asset, all should be non-negative
      K (float)       - strike price of the option, should be non-negative
      T_max (float)   - time untill option expiration date, should be positive
      T (float)       - some time untill expiration, should be non-negative
      r (float)       - risk-free interest rate, should be non-negative
      sigma (float)   - volatility, should be non-negative
      alpha (float)   - degree of the fractional derrivative, should be in [0,1]
      M (int)         - number of time range subdivisions, should be > 3
      N (int)         -number of price range subdivisions, should be > 3
    Returns:
      None
    '''

    if S_max <= 0:
        raise ValueError(
            f"Expected positive number as S_max, got {S_max} instead.")
    minimal_price = np.min(S0)
    if minimal_price < 0:
        raise ValueError(
            f"Expected non-negative numbers as elements of S0, got {minimal_price} instead.")
    if K < 0:
        raise ValueError(
            f"Expected non-negative number as K, got {K} instead.")
    if T_max <= 0:
        raise ValueError(
            f"Expected positive number as T_max, got {T_max} instead.")
    if T < 0:
        raise ValueError(
            f"Expected non-negative number as T, got {T} instead.")
    if r < 0:
        raise ValueError(
            f"Expected non-negative number as r, got {r} instead.")
    if sigma < 0:
        raise ValueError(
            f"Expected non-negative number as sigma, got {sigma} instead.")
    if alpha < 0 or alpha > 1:
        raise ValueError(
            f"Expected a number in range [0, 1] as alpha, got {alpha} instead.")
    if M <= 3:
        raise ValueError(
            f"Expected an integer greater than 3 as M, got {M} instead.")
    if N <= 3:
        raise ValueError(
            f"Expected an integer greater than 3 as N, got {N} instead.")


def AnalyticBlackScholes(S0: np.ndarray,
                         K: float,
                         T: float,
                         r: float,
                         sigma: float,
                         call: bool = True) -> np.ndarray:
    '''
    Calculates the price of european call or put option using the
    Black-Scholes formula.

    Args:
      S0 (np.ndarray) - array of price values of the base asset
      K (float)       - strike price of the option
      T (float)       - time untill option expiration date
      r (float)       - risk-free interest rate
      sigma (float)   - volatility
      call (bool)     - True to get the call option price,
                        False to get put option price

    Returns:
      np.ndarray: numpy array of prices of an option
    '''

    adequacy_checker(S0=S0, K=K, T=T, r=r, sigma=sigma)

    np.seterr(divide='ignore')
    # case T=0 is processed separately to avoid division by zero
    if T == 0:
        if call:
            return np.maximum(S0-K, 0)
        else:
            return np.maximum(K-S0, 0)

    # Black-Scholes formula is used (https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
    d_1 = (np.log(S0 / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * T ** 0.5)
    d_2 = (np.log(S0 / K) + (r - 0.5 * (sigma ** 2)) * T) / (sigma * T ** 0.5)
    if call:
        return S0 * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S0 * norm.cdf(-d_1)


def FBSsolverFDM(S_max: float,
                 K: float,
                 T_max: float,
                 r: float,
                 sigma: float,
                 alpha: float,
                 M: int,
                 N: int,
                 call: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns a matrix of approximate prices of european call or put options, with
    time untill expiration less than T_max and base asset price less than S_max,
    calculated by solving the fractional Black-Scholes equation using the finite
    difference scheme.
    Finite difference scheme is taken from "A robust numerical solution to a
    time-fractional Black–Scholes equation" by S.M. Nuugulu1, F.Gideon & K.C.Patidar.

    Args:
      S_max (float) - maximal base asset price
      K (float)     - strike price of the option
      T_max (float) - maximal time untill option expiration date
      r (float)     - risk-free interest rate
      sigma (float) - volatility
      alpha (float) - degree of fractional derivative
      M (int)       - number of subdivisions for time grid
      N (int)       - number of subdivisions for price grid
      call (bool)   - True to get the call option price,
                      False to get put option price

    Returns:
      np.ndarray: (M+1) * (N+1) Matrix U, with U[i, j] being the price of option with time untill expiration
      t[i] and base asset price S[j]
      np.ndarray: t of length M+1 - time interval [0, T_max] discretized into M equal subintervals
      np.ndarray: S of length N+1 - price interval [0, S_max] discretized into N equal subintervals
    '''

    adequacy_checker(S_max=S_max, K=K, T_max=T_max, r=r,
                     sigma=sigma, alpha=alpha, M=M, N=N)

    # time and price increments
    dt = T_max / M
    dS = S_max / N

    # time and price grids
    t = np.linspace(0, T_max, M+1)
    S = np.linspace(0, S_max, N+1)

    # preparing coefficients for the finite difference
    # scheme in case of a truely fractional equation
    if alpha < 1:
        b = np.zeros(M + 1)
        for j in range(M + 1):
            b[j] = (j + 1) ** (1 - alpha) - j ** (1 - alpha)

    theta = 1 / (gamma(2 - alpha) * (dt ** alpha))

    # preparing the tridiagonal matrix for the scheme
    A = -np.power(S * sigma / dS, 2) / 2 + (r * S) / (2 * dS)
    B = theta + np.power(S * sigma / dS, 2) + r
    C = -np.power(S * sigma / dS, 2) / 2 - (r * S) / (2 * dS)

    upper_diagonal = C[1:-2]
    central_diagonal = B[1:-1]
    lower_diagonal = A[2:-1]

    Matrix = np.zeros((3, central_diagonal.shape[0]))
    Matrix[0, 1:] = upper_diagonal
    Matrix[1, :] = central_diagonal
    Matrix[2, :-1] = lower_diagonal

    # creating the solution matrix and filling in boundary conditions
    U = np.zeros((M + 1, N + 1))
    if call:
        U[0, :] = np.maximum(S - K, 0)
        for i in range(M+1):
            U[i, -1] = AnalyticBlackScholes(S_max, K, t[i], r, sigma, True)
    else:
        U[0, :] = np.maximum(K - S, 0)
        for i in range(M+1):
            U[i, 0] = AnalyticBlackScholes(0, K, t[i], r, sigma, False)
            U[i, -1] = AnalyticBlackScholes(S_max, K, t[i], r, sigma, False)

    # solution matrix is filled row-by-row in time using the implicit scheme
    for j in range(1, M+1):
        # the case of non-fractional derivative is processed separately
        if alpha < 1:
            R = U[0, 1:-1] * b[j-1]
            for k in range(1, j):
                R += (b[j - k - 1] - b[j - k]) * U[k, 1:-1]
            R *= theta
            R[0] -= A[1] * U[j, 0]
            R[-1] -= C[N - 1] * U[j, N]
        else:
            R = U[j-1, 1:-1] * theta
            R[0] -= A[1] * U[j, 0]
            R[-1] -= C[N - 1] * U[j, N]

        U[j, 1:-1] = solve_banded((1, 1), Matrix, R)

    return U, t, S


def FBSextract_value(S0: float,
                     K: float,
                     T0: float,
                     r: float,
                     sigma: float,
                     alpha: float,
                     M: int,
                     N: int,
                     T_max: float = -1,
                     S_max: float = -1,
                     call: bool = True) -> float:
    '''
    Uses FBSsolverFDM to solve the fractional Black-Scholes equation on time
    interval [0, T_max] and price interval [0, S_max], then extracts the option's
    price ONLY at time untill expiration T0 and base asset price S0.
    If T_max and/or S_max are not provided, default boundaries of T_max = T0,
    S_max = 10 * S0 are set.

    Args:
      S0 (float)    - base asset price at which option price will be calculated
      K (float)     - strike price of the option
      T0 (float)    - time untill expiration date at which option price will be calculated
      r (float)     - risk-free interest rate
      sigma (float) - volatility
      alpha (float) - degree of fractional derivative
      M (int)       - number of subdivisions for time grid
      N (int)       - number of subdivisions for price grid
      S_max (float) - maximal base asset price, 10 * S0 by default
      T_max (float) - maximal time untill expiration, T0 by default
      call (bool)   - True to get the call option price,
                      False to get put option price

    Returns:
      (M+1) * (N+1) Matrix U, with U[i, j] being the price of option with time untill expiration
      t[i] and base asset price S[j]
      Array t of length M+1 - time interval [0, T_max] discretized into M equal subintervals
      Array S of length N+1 - price interval [0, S_max] discretized into N equal subintervals
    '''

    adequacy_checker(S0=S0, K=K, T=T0, r=r, sigma=sigma,
                     alpha=alpha, S_max=S_max, T_max=T_max, M=M, N=N)

    U, t, S = FBSsolverFDM(S_max, K, T, r, sigma, alpha, M, N, call)
    idx_t = np.argwhere(np.isclose(t, t_0, atol=0.5 * T/M))[0]
    idx_S = np.argwhere(np.isclose(S, S_0, atol=0.5 * S_max/N))[0]
    return U[idx_t, idx_S][0]
