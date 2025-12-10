import matplotlib.pyplot as plt
import numpy as np
import scipy

np.random.seed(0)


def simdata_settings():
    # Define the dimensions of the input data
    k = 2  # # of oscillators
    n = 4  # # of electrodes
    M = 3  # # of switching states

    # Other parameters
    fs = 100  # Sampling frequency

    # State parameters
    osc_freqs = np.asarray([7, 7])  # Oscillation frequency for each oscillator
    rhos = np.asarray([0.9, 0.9])  # Damping parameter for each oscillator
    var_state_nois = np.asarray([1, 1])  # State noise for each oscillator

    # Observation parameters
    var_obs_noi = 1

    # Model matrices
    x_dim = k * 2
    A, Q = build_AQ(
        M, fs, osc_freqs, rhos, var_state_nois
    )  # Transition matrix and state noise covariance
    R = build_R(n, M, var_obs_noi)  # Observation noise covariance

    B = np.zeros((n, x_dim, M))  # Observation matrix
    B[:, :, 0] = [[0.4, 0, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B[:, :, 1] = [[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, -0.25]]
    B[:, :, 2] = [[0.5, 0, 0, 0], [-0.5, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0.4, 0]]

    Z = np.asarray(
        [[0.998, 0.001, 0.001], [0.001, 0.998, 0.001], [0.001, 0.001, 0.998]]
    )  # Discrete state transition matrix

    X0 = np.random.multivariate_normal(
        np.zeros(x_dim), np.eye(x_dim)
    ).T  # Initial oscillatory state
    S0 = 0  # Initial discrete state

    return (
        fs,
        k,
        n,
        M,
        osc_freqs,
        rhos,
        var_state_nois,
        var_obs_noi,
        A,
        Q,
        R,
        B,
        Z,
        X0,
        S0,
    )


def build_AQ(M, fs, osc_freqs, rhos, var_state_nois):

    k = len(osc_freqs)
    assert len(rhos) == k
    assert len(var_state_nois) == k

    x_dim = k * 2
    A = np.zeros((x_dim, x_dim, M))
    Q = np.zeros((x_dim, x_dim, M))

    for i in range(M):
        oscmats = []
        varmats = []
        for osc_freq, rho, var_state_noi in zip(osc_freqs, rhos, var_state_nois):
            theta = (2 * np.pi * osc_freq) * (1 / fs)
            oscmat = np.asarray(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            oscmats.append(rho * oscmat)
            varmats.append(var_state_noi * np.eye(2))

        A[:, :, i] = scipy.linalg.block_diag(*oscmats)
        Q[:, :, i] = scipy.linalg.block_diag(*varmats)

    return A, Q


def build_R(n, M, var_obs_noi):
    R = np.zeros((n, n, M))
    for i in range(M):
        R[:, :, i] = var_obs_noi * np.eye(n)
    return R


def simulate(A, B0, Q, R, Z, X_0, S_0, T, s=None):
    rng = np.random.default_rng(14)

    n = R.shape[0]  # # of electrodes
    x_dim = A.shape[0]  # # of oscillators*2 == k*2 == continuous hidden state dimension
    M = A.shape[-1]  # # of switching states

    blnSimS = s is None
    if blnSimS:
        s = np.zeros(T, dtype=int)
        s[0] = S_0

    x = np.zeros([T, x_dim])
    x[0, :] = X_0
    y = np.zeros([T, n])
    y[0, :] = B0[:, :, S_0] @ X_0 + rng.multivariate_normal(np.zeros(n), R[:, :, S_0])
    for t in range(1, T):
        if blnSimS:
            s[t] = np.nonzero(rng.multinomial(1, Z[s[t - 1], :]))[0][
                0
            ]  # Save the integer in [0,M-1]
        x[t, :] = A[:, :, s[t]] @ x[t - 1, :] + rng.multivariate_normal(
            np.zeros(x_dim), Q[:, :, s[t]]
        )
        y[t, :] = B0[:, :, s[t]] @ x[t, :] + rng.multivariate_normal(
            np.zeros(n), R[:, :, s[t]]
        )
    return y, s, x


def simulate_model(T: int = 30000, blnSimS: bool = False):
    """

    Parameters
    ----------
    T : int, optional
        _description_, by default 30000
    blnSimS : bool, optional

    Returns
    -------
    fs : int
        Sampling frequency
    k : int
        Number of oscillators
    n : int
        Number of electrodes
    M : int
        Number of switching states
    osc_freqs : np.ndarray, shape (k,)
        Oscillation frequency for each oscillator
    rhos : np.ndarray, shape (k,)
        Damping parameter for each oscillator
    var_state_nois : np.ndarray, shape (k,)
        State noise for each oscillator
    var_obs_noi : int
        Observation noise covariance
    A : np.ndarray, shape (x_dim, x_dim, M)
        Transition matrix
    Q : np.ndarray, shape (x_dim, x_dim, M)
        State noise covariance
    R : np.ndarray, shape (n, n, M)
        Observation noise covariance
    B : np.ndarray, shape (n, x_dim, M)
        Observation matrix
    Z : np.ndarray, shape (M, M)
        Discrete state transition matrix
    X0 : np.ndarray, shape (x_dim,)
        Initial oscillatory state
    S0 : int
        Initial discrete state
    x_dim : int
        Hidden state dimension
    s : np.ndarray, shape (T,)
        Discrete state sequence
    y : np.ndarray, shape (T, n)
        Observation sequence
    x : np.ndarray, shape (T, x_dim)
        Hidden state sequence
    time : np.ndarray, shape (T,)
        Time sequence
    """
    fs, k, n, M, osc_freqs, rhos, var_state_nois, var_obs_noi, A, Q, R, B, Z, X0, S0 = (
        simdata_settings()
    )

    x_dim = k * 2

    # Simulate

    ta = np.arange(T) / fs

    if blnSimS:
        s = None
    else:
        s = np.zeros(T, dtype=int)
        s[ta > 80] = 1
        s[ta > 200] = 2

    y, s, x = simulate(A, B, Q, R, Z, X0, S0, T, s=s)
    time = np.arange(T) / fs

    return (
        fs,
        k,
        n,
        M,
        osc_freqs,
        rhos,
        var_state_nois,
        var_obs_noi,
        A,
        Q,
        R,
        B,
        Z,
        X0,
        S0,
        x_dim,
        s,
        y,
        x,
        time,
    )


def simulate_distinguishable_states(
    n_time: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Generate data where discrete states are easy to distinguish.

    Uses very different dynamics between states:
    - State 0: stable (A=0.5), low noise (Q=0.01)
    - State 1: near unit root (A=0.99), high noise (Q=1.0)
    - Long stays in each state (Z diagonal = 0.98)
    - Low observation noise

    Parameters
    ----------
    n_time : int
        Number of time steps.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with:
        - obs: observations, shape (n_time, n_obs)
        - true_states: discrete state sequence, shape (n_time,)
        - true_continuous: continuous state sequence, shape (n_time, n_cont)
        - params: dict of all model parameters
    """
    rng = np.random.default_rng(seed)

    n_cont = 1
    n_obs = 1
    n_disc = 2

    # Very different dynamics
    A = np.array([[[0.5]], [[0.99]]]).T  # stable vs near unit root
    Q = np.array([[[0.01]], [[1.0]]]).T  # low vs high noise
    H = np.array([[[1.0]], [[1.0]]]).T
    R = np.array([[[0.1]], [[0.1]]]).T  # low observation noise
    Z = np.array([[0.98, 0.02], [0.02, 0.98]])  # long stays

    init_mean = np.zeros((n_cont, n_disc))
    init_cov = np.eye(n_cont)[..., None] * np.ones((1, 1, n_disc))
    init_prob = np.array([0.5, 0.5])

    # Simulate discrete states
    s = np.zeros(n_time, dtype=int)
    s[0] = rng.choice(n_disc, p=init_prob)
    for t in range(1, n_time):
        s[t] = rng.choice(n_disc, p=Z[s[t - 1]])

    # Simulate continuous states
    x = np.zeros((n_time, n_cont))
    x[0] = rng.multivariate_normal(init_mean[:, s[0]], init_cov[:, :, s[0]])
    for t in range(1, n_time):
        w = rng.multivariate_normal(np.zeros(n_cont), Q[:, :, s[t]])
        x[t] = A[:, :, s[t]] @ x[t - 1] + w

    # Simulate observations
    y = np.zeros((n_time, n_obs))
    for t in range(n_time):
        v = rng.multivariate_normal(np.zeros(n_obs), R[:, :, s[t]])
        y[t] = H[:, :, s[t]] @ x[t] + v

    params = {
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "Z": Z,
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "n_cont": n_cont,
        "n_obs": n_obs,
        "n_disc": n_disc,
    }

    return {
        "obs": y,
        "true_states": s,
        "true_continuous": x,
        "params": params,
    }


def simulate_challenging_states(
    n_time: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Generate data where discrete states are harder to distinguish.

    Similar dynamics between states, requiring careful inference:
    - State 0: A=0.85, Q=0.15
    - State 1: A=0.90, Q=0.25
    - Moderate stays (Z diagonal = 0.90)
    - Higher observation noise

    Parameters
    ----------
    n_time : int
        Number of time steps.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with:
        - obs: observations, shape (n_time, n_obs)
        - true_states: discrete state sequence, shape (n_time,)
        - true_continuous: continuous state sequence, shape (n_time, n_cont)
        - params: dict of all model parameters
    """
    rng = np.random.default_rng(seed)

    n_cont = 1
    n_obs = 1
    n_disc = 2

    # Similar dynamics (harder to distinguish)
    A = np.array([[[0.85]], [[0.90]]]).T
    Q = np.array([[[0.15]], [[0.25]]]).T
    H = np.array([[[1.0]], [[1.0]]]).T
    R = np.array([[[0.5]], [[0.5]]]).T  # higher observation noise
    Z = np.array([[0.90, 0.10], [0.10, 0.90]])  # more frequent switching

    init_mean = np.zeros((n_cont, n_disc))
    init_cov = np.eye(n_cont)[..., None] * np.ones((1, 1, n_disc))
    init_prob = np.array([0.5, 0.5])

    # Simulate discrete states
    s = np.zeros(n_time, dtype=int)
    s[0] = rng.choice(n_disc, p=init_prob)
    for t in range(1, n_time):
        s[t] = rng.choice(n_disc, p=Z[s[t - 1]])

    # Simulate continuous states
    x = np.zeros((n_time, n_cont))
    x[0] = rng.multivariate_normal(init_mean[:, s[0]], init_cov[:, :, s[0]])
    for t in range(1, n_time):
        w = rng.multivariate_normal(np.zeros(n_cont), Q[:, :, s[t]])
        x[t] = A[:, :, s[t]] @ x[t - 1] + w

    # Simulate observations
    y = np.zeros((n_time, n_obs))
    for t in range(n_time):
        v = rng.multivariate_normal(np.zeros(n_obs), R[:, :, s[t]])
        y[t] = H[:, :, s[t]] @ x[t] + v

    params = {
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "Z": Z,
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "n_cont": n_cont,
        "n_obs": n_obs,
        "n_disc": n_disc,
    }

    return {
        "obs": y,
        "true_states": s,
        "true_continuous": x,
        "params": params,
    }
