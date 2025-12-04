"""
LQR Trajectory Tracking Script for Planar Quadrotor

This script demonstrates LQR control for tracking a time-varying reference
trajectory. The reference changes the horizontal position x at different
time intervals while maintaining a constant altitude.

Usage:
    python main_tracking.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode
from code.controllers import design_lqr_controller


def reference(t, z_ref=0.5):
    """
    Returns the desired state x_ref(t) = [x_ref, xdot_ref, z_ref, zdot_ref, theta_ref, thetadot_ref].

    The reference trajectory:
        - for t < 2.0: x_ref = 0.0
        - for 2.0 <= t < 5.0: x_ref = 0.5
        - for t >= 5.0: x_ref = 0.0
        - Always: z_ref = 0.5, theta_ref = 0.0, all velocities = 0

    Parameters
    ----------
    t : float or ndarray
        Current time or array of times
    z_ref : float, optional
        Desired vertical position, default 0.5

    Returns
    -------
    x_ref : ndarray, shape (6,) or (N, 6)
        Reference state vector [x_ref, xdot_ref, z_ref, zdot_ref, theta_ref, thetadot_ref]
    """
    t = np.array(t)
    is_scalar = t.ndim == 0
    if is_scalar:
        t = np.array([t])
    
    # Initialize reference state array
    x_ref = np.zeros((len(t), 6))
    
    # Set z_ref for all times
    x_ref[:, 2] = z_ref
    
    # Set x_ref based on time intervals
    mask1 = t < 2.0
    mask2 = (t >= 2.0) & (t < 5.0)
    mask3 = t >= 5.0
    
    x_ref[mask1, 0] = 0.0
    x_ref[mask2, 0] = 0.5
    x_ref[mask3, 0] = 0.0
    
    # All velocities and theta are zero (already initialized)
    
    if is_scalar:
        return x_ref[0]
    return x_ref


def main():
    """
    Main function for LQR trajectory tracking simulation.

    This function:
        1. Defines system parameters (same as main_lqr.py)
        2. Designs LQR controller (reuses design_lqr_controller)
        3. Defines reference trajectory function
        4. Implements LQR tracking control law
        5. Runs simulation from perturbed initial condition
        6. Plots results showing tracking performance
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
    print("LQR TRAJECTORY TRACKING - PLANAR QUADROTOR")
    print("=" * 60)

    # Design LQR controller (reuse from controllers.py)
    print("\nDesigning LQR controller...")
    K, A, B = design_lqr_controller(m=m, I=I, l=l, g=g)
    print(f"LQR gain matrix K shape: {K.shape}")

    # Define hover control input
    u_hover = np.array([m * g, 0.0])

    # Define LQR tracking control law
    def u_func(t, state):
        """
        LQR tracking control law: u = u_hover + u_tilde
        where u_tilde = -K @ (state - x_ref)
        
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
        x_ref = reference(t, z_ref=0.5)
        x_tilde = state - x_ref
        u_tilde = -K @ x_tilde
        u = u_hover + u_tilde
        return u

    # Initial conditions (perturbed from initial reference)
    state0 = np.array([0.2, 0.0, 0.8, 0.0, 0.15, 0.0])
    t_span = (0.0, 8.0)
    dt = 0.01

    print(f"\nInitial state: x0 = {state0}")
    print(f"Reference at t=0: x_ref(0) = {reference(0.0)}")
    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(state0, u_func, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"Final reference: x_ref({t[-1]:.2f}) = {reference(t[-1])}")

    # Compute reference trajectory for plotting
    x_ref_traj = np.array([reference(ti) for ti in t])

    # Plot results
    print("\nGenerating plots...")

    # Get project root directory (parent of code directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # Figure 1: x(t) and z(t) with reference trajectories
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot x(t) and x_ref(t)
    axes1[0].plot(t, states[:, 0], 'b-', linewidth=2, label='x(t)')
    axes1[0].plot(t, x_ref_traj[:, 0], 'r--', linewidth=2, label='x_ref(t)')
    axes1[0].set_xlabel('Time (s)', fontsize=12)
    axes1[0].set_ylabel('x (m)', fontsize=12)
    axes1[0].set_title('LQR Tracking – Horizontal Position', fontsize=14)
    axes1[0].legend(fontsize=11)
    axes1[0].grid(True, alpha=0.3)
    
    # Plot z(t) and z_ref(t)
    axes1[1].plot(t, states[:, 2], 'b-', linewidth=2, label='z(t)')
    axes1[1].plot(t, x_ref_traj[:, 2], 'r--', linewidth=2, label='z_ref(t)')
    axes1[1].set_xlabel('Time (s)', fontsize=12)
    axes1[1].set_ylabel('z (m)', fontsize=12)
    axes1[1].set_title('LQR Tracking – Vertical Position', fontsize=14)
    axes1[1].legend(fontsize=11)
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tracking_xz_path = os.path.join(visuals_dir, 'tracking_xz.png')
    plt.savefig(tracking_xz_path, dpi=150, bbox_inches='tight')
    print(f"Saved tracking plot to {tracking_xz_path}")

    # Figure 2: theta(t) versus reference (0)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(t, states[:, 4], 'b-', linewidth=2, label='theta(t)')
    ax2.axhline(y=0.0, color='r', linestyle='--', linewidth=2, label='theta_ref = 0')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('theta (rad)', fontsize=12)
    ax2.set_title('LQR Tracking – Pitch Angle', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    tracking_theta_path = os.path.join(visuals_dir, 'tracking_theta.png')
    plt.savefig(tracking_theta_path, dpi=150, bbox_inches='tight')
    print(f"Saved theta plot to {tracking_theta_path}")

    # Show plots
    plt.show()

    print("\n" + "=" * 60)
    print("LQR trajectory tracking simulation complete!")
    print("=" * 60)
    print("\nThe plots show that the LQR controller successfully tracks")
    print("the time-varying reference trajectory, with x_ref changing")
    print("at t=2.0s and t=5.0s while maintaining z_ref=0.5 and theta_ref=0.")


if __name__ == "__main__":
    main()

