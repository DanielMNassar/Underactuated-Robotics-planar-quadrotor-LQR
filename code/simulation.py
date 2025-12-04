"""
Simulation Module for Planar Quadrotor

This module provides simulation engines for time integration of the
planar quadrotor dynamics. Supports multiple integration methods.

Functions:
    linearize_quadrotor: Compute linearized A and B matrices around hover
    simulate_euler: Euler integration method
    simulate_rk4: Runge-Kutta 4th order method
    simulate_ode: Using scipy.integrate.solve_ivp
"""

import numpy as np
from scipy.integrate import solve_ivp


def linearize_quadrotor(m=1.0, I=0.02, l=0.25, g=9.81):
    """
    Returns A, B matrices for the planar quadrotor linearized around hover.

    The linearization is performed around the hover equilibrium:
        - theta = 0 (no pitch)
        - xdot = 0, zdot = 0, thetadot = 0 (no velocities)
        - FT = m*g (thrust balances gravity)
        - tau = 0 (no torque)
        - x* = [0, 0, 0, 0, 0, 0]
        - u* = [m*g, 0]

    The Jacobians are computed analytically from the nonlinear dynamics:
        m * xddot      = -FT * sin(theta)
        m * zddot      =  FT * cos(theta) - m*g
        I * thetaddot  = tau

    Parameters
    ----------
    m : float, optional
        Mass (kg), default 1.0
    I : float, optional
        Moment of inertia (kg⋅m²), default 0.02
    l : float, optional
        Half-distance between rotors (m), default 0.25
        Note: l is not used in linearization but kept for consistency
    g : float, optional
        Gravitational acceleration (m/s²), default 9.81

    Returns
    -------
    A : ndarray, shape (6, 6)
        State matrix of the linearized system
        A = ∂f/∂x evaluated at hover equilibrium
    B : ndarray, shape (6, 2)
        Input matrix of the linearized system
        B = ∂f/∂u evaluated at hover equilibrium

    Notes
    -----
    The state vector is: [x, xdot, z, zdot, theta, thetadot]
    The input vector is: [FT, tau]

    The state derivative function is:
        f(x, u) = [xdot, -FT*sin(theta)/m, zdot, (FT*cos(theta)-m*g)/m, thetadot, tau/I]

    At equilibrium (theta=0, FT=m*g, tau=0):
        - ∂f₂/∂theta = -FT*cos(theta)/m = -g
        - ∂f₄/∂FT = cos(theta)/m = 1/m
        - ∂f₆/∂tau = 1/I

    Examples
    --------
    >>> A, B = linearize_quadrotor(m=1.0, I=0.02, l=0.25, g=9.81)
    >>> print(f"A shape: {A.shape}, B shape: {B.shape}")
    A shape: (6, 6), B shape: (6, 2)
    """
    # Initialize A and B matrices
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))

    # State vector: [x, xdot, z, zdot, theta, thetadot]
    # Input vector: [FT, tau]

    # State derivative function: f(x, u) = [xdot, -FT*sin(theta)/m, zdot, (FT*cos(theta)-m*g)/m, thetadot, tau/I]

    # Compute A = ∂f/∂x (evaluated at hover: theta=0, FT=m*g, tau=0)
    # Row 0: ∂f₁/∂x = ∂(xdot)/∂x = [0, 1, 0, 0, 0, 0]
    A[0, 1] = 1.0

    # Row 1: ∂f₂/∂x = ∂(-FT*sin(theta)/m)/∂x
    # At equilibrium: theta=0, so sin(0)=0, but we need ∂f₂/∂theta
    # ∂f₂/∂theta = -FT*cos(theta)/m
    # At theta=0: -FT*cos(0)/m = -FT/m = -g (since FT = m*g at equilibrium)
    A[1, 4] = -g

    # Row 2: ∂f₃/∂x = ∂(zdot)/∂x = [0, 0, 0, 1, 0, 0]
    A[2, 3] = 1.0

    # Row 3: ∂f₄/∂x = ∂((FT*cos(theta)-m*g)/m)/∂x
    # At equilibrium: theta=0, so ∂f₄/∂theta = -FT*sin(theta)/m = 0
    # (No dependence on theta at equilibrium for zddot)
    A[3, 4] = 0.0  # This is zero at equilibrium

    # Row 4: ∂f₅/∂x = ∂(thetadot)/∂x = [0, 0, 0, 0, 0, 1]
    A[4, 5] = 1.0

    # Row 5: ∂f₆/∂x = ∂(tau/I)/∂x = [0, 0, 0, 0, 0, 0]
    # (No state dependence)

    # Compute B = ∂f/∂u (evaluated at hover: theta=0, FT=m*g, tau=0)
    # Row 0: ∂f₁/∂u = ∂(xdot)/∂u = [0, 0]
    # (No input dependence)

    # Row 1: ∂f₂/∂u = ∂(-FT*sin(theta)/m)/∂u
    # ∂f₂/∂FT = -sin(theta)/m
    # At theta=0: -sin(0)/m = 0
    B[1, 0] = 0.0

    # Row 2: ∂f₃/∂u = ∂(zdot)/∂u = [0, 0]
    # (No input dependence)

    # Row 3: ∂f₄/∂u = ∂((FT*cos(theta)-m*g)/m)/∂u
    # ∂f₄/∂FT = cos(theta)/m
    # At theta=0: cos(0)/m = 1/m
    B[3, 0] = 1.0 / m

    # Row 4: ∂f₅/∂u = ∂(thetadot)/∂u = [0, 0]
    # (No input dependence)

    # Row 5: ∂f₆/∂u = ∂(tau/I)/∂u
    # ∂f₆/∂tau = 1/I
    B[5, 1] = 1.0 / I

    return A, B


def simulate_euler(state0, u_func, params, t_span, dt):
    """
    Simulate the system using Euler integration method.

    Parameters
    ----------
    state0 : array_like, shape (6,)
        Initial state vector [x, xdot, z, zdot, theta, thetadot]
    u_func : callable
        Control function u(t) that returns control input [FT, tau]
        Signature: u_func(t) -> array_like, shape (2,)
    params : dict
        System parameters dictionary
    t_span : tuple
        Time span (t_start, t_end)
    dt : float
        Time step size

    Returns
    -------
    t : ndarray
        Time array
    states : ndarray, shape (N, 6)
        State trajectory over time
    inputs : ndarray, shape (N, 2)
        Control input trajectory over time

    Notes
    -----
    This function will be implemented in a future task.
    """
    pass


def simulate_rk4(state0, u_func, params, t_span, dt):
    """
    Simulate the system using Runge-Kutta 4th order method.

    Parameters
    ----------
    state0 : array_like, shape (6,)
        Initial state vector [x, xdot, z, zdot, theta, thetadot]
    u_func : callable
        Control function u(t) that returns control input [FT, tau]
        Signature: u_func(t) -> array_like, shape (2,)
    params : dict
        System parameters dictionary
    t_span : tuple
        Time span (t_start, t_end)
    dt : float
        Time step size

    Returns
    -------
    t : ndarray
        Time array
    states : ndarray, shape (N, 6)
        State trajectory over time
    inputs : ndarray, shape (N, 2)
        Control input trajectory over time

    Notes
    -----
    This function will be implemented in a future task.
    """
    pass


def simulate_ode(state0, u_func, params, t_span, dt=None, method='RK45'):
    """
    Simulate the system using scipy.integrate.solve_ivp.

    Parameters
    ----------
    state0 : array_like, shape (6,)
        Initial state vector [x, xdot, z, zdot, theta, thetadot]
    u_func : callable
        Control function that returns control input [FT, tau].
        Can be either:
        - Open-loop: u_func(t) -> array_like, shape (2,)
        - Closed-loop: u_func(t, state) -> array_like, shape (2,)
    params : dict
        System parameters dictionary with keys: 'm', 'I', 'l', 'g'
    t_span : tuple
        Time span (t_start, t_end)
    dt : float, optional
        Desired time step for output. If None, uses solver's default.
        If provided, uses dense_output to evaluate at regular intervals.
    method : str, optional
        Integration method for solve_ivp (default: 'RK45')

    Returns
    -------
    t : ndarray
        Time array
    states : ndarray, shape (N, 6)
        State trajectory over time
    inputs : ndarray, shape (N, 2)
        Control input trajectory over time

    Notes
    -----
    The function automatically detects whether u_func is open-loop or closed-loop
    by checking the number of arguments it accepts.

    Examples
    --------
    >>> # Open-loop control
    >>> def u_openloop(t):
    ...     return np.array([9.81, 0.0])
    >>> t, states, inputs = simulate_ode(state0, u_openloop, params, (0, 10))
    
    >>> # Closed-loop control
    >>> def u_closedloop(t, state):
    ...     return np.array([9.81, 0.0]) - K @ (state - state_ref)
    >>> t, states, inputs = simulate_ode(state0, u_closedloop, params, (0, 10))
    """
    from code.dynamics import compute_derivatives
    import inspect

    # Check if u_func is open-loop (1 arg) or closed-loop (2 args)
    sig = inspect.signature(u_func)
    num_args = len(sig.parameters)
    is_closed_loop = num_args >= 2

    def dynamics_wrapper(t, state):
        """
        Wrapper function for solve_ivp that computes state derivatives.
        Handles both open-loop and closed-loop control.
        """
        # Get control input
        if is_closed_loop:
            u = u_func(t, state)
        else:
            u = u_func(t)

        # Compute state derivatives
        state_dot = compute_derivatives(state, u, params)
        return state_dot

    # Solve the ODE
    sol = solve_ivp(
        dynamics_wrapper,
        t_span,
        state0,
        method=method,
        dense_output=(dt is not None),
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Generate time array and evaluate solution
    if dt is not None:
        # Use dense output to evaluate at regular intervals
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        states = sol.sol(t).T  # Transpose to get (N, 6) shape
    else:
        # Use solver's time points
        t = sol.t
        states = sol.y.T  # Transpose to get (N, 6) shape

    # Compute control inputs at each time step
    inputs = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        if is_closed_loop:
            inputs[i] = u_func(ti, states[i])
        else:
            inputs[i] = u_func(ti)

    return t, states, inputs

