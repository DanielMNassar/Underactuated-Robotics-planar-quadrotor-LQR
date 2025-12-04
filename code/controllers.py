"""
Control Module for Planar Quadrotor

This module implements control algorithms for the planar quadrotor system.

Functions:
    design_lqr_controller: Design LQR controller for planar quadrotor
    linearize: Compute linearized A and B matrices around an equilibrium point
    compute_lqr: Compute LQR gain matrix K
    lqr_controller: LQR feedback control law
"""

import numpy as np
from scipy.linalg import solve_continuous_are

# Try to import python-control, fall back to scipy if not available
try:
    import control
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False


def design_lqr_controller(Q=None, R=None, m=1.0, I=0.02, l=0.25, g=9.81):
    """
    Computes the LQR gain K for the linearized planar quadrotor around hover.

    Uses linearize_quadrotor() to get A, B, then calls an LQR solver.
    The LQR controller minimizes the cost function:
        J = ∫(x^T Q x + u^T R u) dt

    Parameters
    ----------
    Q : ndarray, shape (6, 6), optional
        State weighting matrix (positive semi-definite).
        If None, uses default diagonal matrix that penalizes z, theta, and their velocities.
    R : ndarray, shape (2, 2), optional
        Input weighting matrix (positive definite).
        If None, uses default diagonal matrix.
    m : float, optional
        Mass (kg), default 1.0
    I : float, optional
        Moment of inertia (kg⋅m²), default 0.02
    l : float, optional
        Half-distance between rotors (m), default 0.25
    g : float, optional
        Gravitational acceleration (m/s²), default 9.81

    Returns
    -------
    K : ndarray, shape (2, 6)
        LQR gain matrix. Control law: u = u* - K @ (x - x*)
    A : ndarray, shape (6, 6)
        State matrix of the linearized system
    B : ndarray, shape (6, 2)
        Input matrix of the linearized system

    Notes
    -----
    The function uses python-control's lqr() if available, otherwise
    solves the continuous-time algebraic Riccati equation (CARE) using scipy.

    Default Q matrix penalizes:
        - z position and zdot velocity (vertical control)
        - theta angle and thetadot angular velocity (attitude control)
        - Less penalty on x position and xdot (horizontal drift acceptable)

    Examples
    --------
    >>> K, A, B = design_lqr_controller()
    >>> print(f"LQR gain shape: {K.shape}")
    LQR gain shape: (2, 6)
    """
    # Import linearize_quadrotor from simulation module
    # Handle both relative and absolute imports
    try:
        from .simulation import linearize_quadrotor
    except ImportError:
        # Fall back to absolute import if relative doesn't work
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from code.simulation import linearize_quadrotor

    # Get linearized matrices
    A, B = linearize_quadrotor(m=m, I=I, l=l, g=g)

    # Default Q matrix: penalize z, theta, and their velocities more than x
    if Q is None:
        Q = np.diag([1.0,    # x position (low penalty)
                     0.1,    # xdot velocity (low penalty)
                     10.0,   # z position (high penalty - maintain altitude)
                     5.0,    # zdot velocity (medium penalty)
                     100.0,  # theta angle (very high penalty - maintain attitude)
                     10.0])  # thetadot angular velocity (high penalty)

    # Default R matrix: penalize control inputs
    if R is None:
        R = np.diag([1.0,   # FT thrust (low penalty)
                     1.0])  # tau torque (low penalty)

    # Solve for LQR gain
    if HAS_CONTROL:
        # Use python-control library
        K, S, E = control.lqr(A, B, Q, R)
        # control.lqr returns K in the form u = -K @ x
        # We want u = u* - K @ (x - x*), so we keep K as is
    else:
        # Solve continuous-time algebraic Riccati equation (CARE) manually
        # A^T P + P A - P B R^(-1) B^T P + Q = 0
        # Then K = R^(-1) B^T P
        P = solve_continuous_are(A, B, Q, R)
        R_inv = np.linalg.inv(R)
        K = R_inv @ B.T @ P
        # scipy returns K such that u = -K @ x
        # We want u = u* - K @ (x - x*), so we keep K as is

    return K, A, B


def linearize(state_eq, u_eq, params):
    """
    Linearize the nonlinear dynamics around an equilibrium point.

    Computes the Jacobian matrices A and B:
        A = ∂f/∂x (evaluated at equilibrium)
        B = ∂f/∂u (evaluated at equilibrium)

    Parameters
    ----------
    state_eq : array_like, shape (6,)
        Equilibrium state vector [x, xdot, z, zdot, theta, thetadot]
    u_eq : array_like, shape (2,)
        Equilibrium control input [FT, tau]
    params : dict
        System parameters dictionary

    Returns
    -------
    A : ndarray, shape (6, 6)
        State matrix of the linearized system
    B : ndarray, shape (6, 2)
        Input matrix of the linearized system

    Notes
    -----
    This function will be implemented in a future task.
    The linearization is done around hover condition:
        theta = 0, FT = m*g, tau = 0
    """
    pass


def compute_lqr(A, B, Q, R):
    """
    Compute LQR gain matrix K using python-control.

    Solves the continuous-time algebraic Riccati equation (CARE):
        A^T P + P A - P B R^(-1) B^T P + Q = 0
    Then computes: K = R^(-1) B^T P

    Parameters
    ----------
    A : ndarray, shape (6, 6)
        State matrix
    B : ndarray, shape (6, 2)
        Input matrix
    Q : ndarray, shape (6, 6)
        State weighting matrix (positive semi-definite)
    R : ndarray, shape (2, 2)
        Input weighting matrix (positive definite)

    Returns
    -------
    K : ndarray, shape (2, 6)
        LQR gain matrix
    P : ndarray, shape (6, 6)
        Solution to the Riccati equation
    eigvals : ndarray
        Eigenvalues of the closed-loop system (A - B*K)

    Notes
    -----
    This function will be implemented in a future task.
    Uses python-control library for solving the CARE.
    """
    pass


def lqr_controller(state, state_ref, u_ref, K):
    """
    Compute LQR control input using feedback law.

    Control law: u = u_ref - K * (state - state_ref)

    Parameters
    ----------
    state : array_like, shape (6,)
        Current state vector
    state_ref : array_like, shape (6,)
        Reference state vector
    u_ref : array_like, shape (2,)
        Reference control input (feedforward)
    K : ndarray, shape (2, 6)
        LQR gain matrix

    Returns
    -------
    u : ndarray, shape (2,)
        Control input [FT, tau]

    Notes
    -----
    This function will be implemented in a future task.
    """
    pass

