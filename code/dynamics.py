"""
Nonlinear Dynamics Module for Planar Quadrotor

This module implements the nonlinear dynamics of a 2D quadrotor (planar VTOL).
The system has 6 states and 2 control inputs.

State vector (always use this order):
    state = [x, xdot, z, zdot, theta, thetadot]
    where:
        x        = horizontal position (m)
        xdot     = horizontal velocity (m/s)
        z        = vertical position (m)
        zdot     = vertical velocity (m/s)
        theta    = pitch angle (radians)
        thetadot = angular velocity (rad/s)

Control input vector:
    u = [FT, tau]
    where:
        FT  = total thrust = F1 + F2 (N)
        tau = torque = l * (F2 - F1) (N⋅m)

System parameters:
    m   = mass (kg)
    I   = moment of inertia (kg⋅m²)
    l   = half-distance between rotors (m)
    g   = gravitational acceleration (m/s²)

Nonlinear dynamics:
    m * xddot      = -FT * sin(theta)
    m * zddot      =  FT * cos(theta) - m*g
    I * thetaddot  = tau
"""

import numpy as np


def compute_derivatives(state, u, params):
    """
    Compute the time derivatives of the state vector.

    This function implements the nonlinear dynamics of the planar quadrotor.
    The state derivatives are computed from the current state and control inputs.

    Parameters
    ----------
    state : array_like, shape (6,)
        Current state vector [x, xdot, z, zdot, theta, thetadot]
    u : array_like, shape (2,)
        Control input vector [FT, tau]
    params : dict
        Dictionary containing system parameters:
            - 'm' : mass (kg)
            - 'I' : moment of inertia (kg⋅m²)
            - 'l' : half-distance between rotors (m)
            - 'g' : gravitational acceleration (m/s²), default 9.81

    Returns
    -------
    state_dot : ndarray, shape (6,)
        Time derivatives of the state vector [xdot, xddot, zdot, zddot, thetadot, thetaddot]

    Notes
    -----
    The dynamics are derived from Newton's laws:
    - Horizontal force: m * xddot = -FT * sin(theta)
    - Vertical force: m * zddot = FT * cos(theta) - m*g
    - Rotational dynamics: I * thetaddot = tau

    Examples
    --------
    >>> state = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    >>> u = [9.81, 0.0]  # Hover condition
    >>> params = {'m': 1.0, 'I': 0.1, 'l': 0.2, 'g': 9.81}
    >>> state_dot = compute_derivatives(state, u, params)
    """
    # Extract state variables
    x, xdot, z, zdot, theta, thetadot = state

    # Extract control inputs
    FT, tau = u

    # Extract parameters
    m = params['m']
    I = params['I']
    g = params.get('g', 9.81)  # Default to 9.81 if not specified

    # Compute accelerations from nonlinear dynamics
    # Horizontal acceleration: m * xddot = -FT * sin(theta)
    xddot = -FT * np.sin(theta) / m

    # Vertical acceleration: m * zddot = FT * cos(theta) - m*g
    zddot = (FT * np.cos(theta) - m * g) / m

    # Angular acceleration: I * thetaddot = tau
    thetaddot = tau / I

    # Construct state derivative vector
    # [xdot, xddot, zdot, zddot, thetadot, thetaddot]
    state_dot = np.array([
        xdot,        # derivative of x
        xddot,       # derivative of xdot
        zdot,        # derivative of z
        zddot,       # derivative of zdot
        thetadot,    # derivative of theta
        thetaddot    # derivative of thetadot
    ])

    return state_dot


def get_hover_condition(params):
    """
    Get the hover equilibrium condition for the planar quadrotor.

    Hover condition:
        - theta = 0 (no pitch)
        - FT = m*g (thrust balances gravity)
        - tau = 0 (no torque)
        - All velocities and positions can be set to zero for reference

    Parameters
    ----------
    params : dict
        Dictionary containing system parameters:
            - 'm' : mass (kg)
            - 'g' : gravitational acceleration (m/s²), default 9.81

    Returns
    -------
    state_hover : ndarray, shape (6,)
        Hover state vector [0, 0, 0, 0, 0, 0]
    u_hover : ndarray, shape (2,)
        Hover control input [m*g, 0]

    Examples
    --------
    >>> params = {'m': 1.0, 'I': 0.1, 'l': 0.2, 'g': 9.81}
    >>> state_hover, u_hover = get_hover_condition(params)
    >>> print(f"Hover thrust: {u_hover[0]:.2f} N")
    Hover thrust: 9.81 N
    """
    m = params['m']
    g = params.get('g', 9.81)

    # Hover state: all zeros (can be shifted to any position)
    state_hover = np.zeros(6)

    # Hover control: thrust balances gravity, no torque
    u_hover = np.array([m * g, 0.0])

    return state_hover, u_hover

