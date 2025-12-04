"""
LQR Controlled Simulation with Wind Disturbance

This script runs a closed-loop simulation of the planar quadrotor
using LQR control with an external horizontal wind disturbance.

The wind disturbance is unknown to the controller, demonstrating
how the LQR controller handles unmodeled disturbances.

Usage:
    python main_lqr_wind.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import linearize_quadrotor
from code.visualization import plot_states, plot_inputs, animate_quadrotor
import control


def wind_force(t):
    """
    External horizontal disturbance force F_wind(t) [N].

    For now:
      - 0 N for t < 2 s
      - 0.5 N for t >= 2 s
    """
    if t < 2.0:
        return 0.0
    else:
        return 0.5


def quadrotor_dynamics_with_wind(t, state, u_func, params):
    """
    Nonlinear planar quadrotor dynamics with added horizontal disturbance.

    State x = [x, xdot, z, zdot, theta, thetadot]
    Inputs u = [FT, tau]

    m * xddot     = -FT * sin(theta) + F_wind(t)
    m * zddot     =  FT * cos(theta) - m*g
    I * thetaddot = tau

    Parameters
    ----------
    t : float
        Current time
    state : array_like, shape (6,)
        Current state vector [x, xdot, z, zdot, theta, thetadot]
    u_func : callable
        Control function u_func(t, state) -> [FT, tau]
    params : dict
        System parameters dictionary with keys: 'm', 'I', 'l', 'g'

    Returns
    -------
    state_dot : ndarray, shape (6,)
        Time derivatives of the state vector
    """
    # Extract state variables
    x, xdot, z, zdot, theta, thetadot = state

    # Get control input
    u = u_func(t, state)
    FT, tau = u

    # Extract parameters
    m = params['m']
    I = params['I']
    g = params.get('g', 9.81)

    # Get wind force
    F_wind = wind_force(t)

    # Compute accelerations from nonlinear dynamics with wind
    # Horizontal acceleration: m * xddot = -FT * sin(theta) + F_wind(t)
    xddot = (-FT * np.sin(theta) + F_wind) / m

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


def simulate_with_wind(state0, u_func, params, t_span, dt):
    """
    Use scipy.integrate.solve_ivp with t_eval and return:
      t: (N,)
      states: (N, 6)
      inputs: (N, 2)  # reconstructed using u_func(t, x)

    Parameters
    ----------
    state0 : array_like, shape (6,)
        Initial state vector [x, xdot, z, zdot, theta, thetadot]
    u_func : callable
        Control function u_func(t, state) -> [FT, tau]
    params : dict
        System parameters dictionary with keys: 'm', 'I', 'l', 'g'
    t_span : tuple
        Time span (t_start, t_end)
    dt : float
        Time step for output

    Returns
    -------
    t : ndarray
        Time array
    states : ndarray, shape (N, 6)
        State trajectory over time
    inputs : ndarray, shape (N, 2)
        Control input trajectory over time
    """
    # Create time evaluation points
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    # Wrapper function for solve_ivp
    def dynamics_wrapper(t, state):
        """Wrapper function for solve_ivp that includes wind."""
        return quadrotor_dynamics_with_wind(t, state, u_func, params)

    # Solve the ODE
    sol = solve_ivp(
        dynamics_wrapper,
        t_span,
        state0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Extract results
    t = sol.t
    states = sol.y.T  # Transpose to get (N, 6) shape

    # Compute control inputs at each time step
    inputs = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        inputs[i] = u_func(ti, states[i])

    return t, states, inputs


def main():
    """
    Main function for LQR-controlled simulation with wind disturbance.

    This function:
        1. Defines system parameters
        2. Designs LQR controller (same as main_lqr.py)
        3. Sets equilibrium state and initial conditions
        4. Runs controlled simulation with wind disturbance
        5. Plots results showing controller response to disturbance
    """
    # System parameters (same as main_lqr.py)
    m = 1.0
    I = 0.02
    l = 0.25
    g = 9.81

    params = {
        'm': m,
        'I': I,
        'l': l,
        'g': g
    }

    print("=" * 60)
    print("LQR-CONTROLLED PLANAR QUADROTOR WITH WIND DISTURBANCE")
    print("=" * 60)

    # Design LQR controller (same as main_lqr.py)
    print("\nDesigning LQR controller...")
    
    # Compute A, B using linearize_quadrotor
    A, B = linearize_quadrotor(m=m, I=I, l=l, g=g)
    
    # Define Q and R exactly as in main_lqr.py
    Q = np.diag([1.0,    # x position (low penalty)
                 0.1,    # xdot velocity (low penalty)
                 10.0,   # z position (high penalty - maintain altitude)
                 5.0,    # zdot velocity (medium penalty)
                 100.0,  # theta angle (very high penalty - maintain attitude)
                 10.0])  # thetadot angular velocity (high penalty)

    R = np.diag([1.0,   # FT thrust (low penalty)
                 1.0])  # tau torque (low penalty)

    # Solve LQR
    K, P, eigvals = control.lqr(A, B, Q, R)
    print(f"LQR gain matrix K shape: {K.shape}")
    print(f"K = \n{K}")

    # Define equilibrium state: x* = [0, 0, z*, 0, 0, 0]
    z_star = 0.5  # Desired altitude
    x_ref = np.array([0.0, 0.0, z_star, 0.0, 0.0, 0.0])
    u_hover = np.array([m * g, 0.0])  # Hover thrust

    print(f"\nEquilibrium state: x_ref = {x_ref}")
    print(f"Equilibrium input: u_hover = {u_hover}")

    # Initial conditions (same as main_lqr.py)
    state0 = np.array([0.2, 0.0, 0.8, 0.0, 0.15, 0.0])  # Perturbed initial condition

    print(f"\nInitial state: x0 = {state0}")
    print(f"Initial error: x0 - x_ref = {state0 - x_ref}")

    # Define control law: u(t, x) = u_hover - K @ (x - x_ref)
    # Note: Controller is unaware of wind disturbance
    def u_func(t, state):
        """
        LQR control law: u = u_hover - K @ (x - x_ref)
        
        Parameters
        ----------
        t : float
            Current time
        state : array_like, shape (6,)
            Current state vector
            
        Returns
        -------
        u : ndarray, shape (2,)
            Control input [FT, tau]
        """
        state = np.array(state)
        x_tilde = state - x_ref
        u_tilde = -K @ x_tilde
        return u_hover + u_tilde

    # Time span and simulation parameters
    t_span = (0.0, 8.0)
    dt = 0.01

    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Wind disturbance: F_wind = 0 N for t < 2s, F_wind = 0.5 N for t >= 2s")
    print("Running simulation...")

    # Run simulation with wind
    t, states, inputs = simulate_with_wind(state0, u_func, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"Final error: x_final - x_ref = {states[-1] - x_ref}")

    # Plot results
    print("\nGenerating plots...")

    # Get project root directory (parent of code directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # Plot states: x(t), z(t), theta(t) (and optionally others)
    fig_states, axes_states = plot_states(t, states)
    plt.suptitle('LQR with Wind Disturbance - State Trajectories', fontsize=14)
    
    # Save state plot
    states_path = os.path.join(visuals_dir, 'lqr_wind_states.png')
    plt.savefig(states_path, dpi=150, bbox_inches='tight')
    print(f"Saved state plot to {states_path}")

    # Plot control inputs
    fig_inputs, axes_inputs = plot_inputs(t, inputs)
    plt.suptitle('LQR with Wind Disturbance - Control Inputs', fontsize=14)
    
    # Save input plot
    inputs_path = os.path.join(visuals_dir, 'lqr_wind_inputs.png')
    plt.savefig(inputs_path, dpi=150, bbox_inches='tight')
    print(f"Saved input plot to {inputs_path}")

    # Create animation
    print("\nCreating animation...")
    animation_path = os.path.join(visuals_dir, 'lqr_quadrotor_wind.mp4')
    anim = animate_quadrotor(t, states, params, save_path=animation_path, fps=30)
    print(f"Animation saved to {animation_path}")

    # Show plots and animation
    # Note: The animation object must be stored in a variable to prevent garbage collection
    plt.show()

    print("\n" + "=" * 60)
    print("LQR simulation with wind disturbance complete!")
    print("=" * 60)
    print("\nThe plots show how the LQR controller responds to the unknown")
    print("wind disturbance. The controller attempts to maintain the desired")
    print("equilibrium state despite the external force.")


if __name__ == "__main__":
    main()

