"""
LQR Smooth Trajectory Tracking Script for Planar Quadrotor

This script demonstrates LQR control for tracking a smooth time-varying reference
trajectory. The reference smoothly transitions the horizontal position x from 0 to 0.5
while maintaining a constant altitude.

Usage:
    python main_tracking_smooth.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Allow running from inside code/ with `python main_tracking_smooth.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode, linearize_quadrotor
from code.visualization import animate_quadrotor
import control  # python-control


def reference_trajectory(t):
    """
    Time-varying reference state:
    [x_ref, xdot_ref, z_ref, zdot_ref, theta_ref, thetadot_ref].

    Scenario:
      - t < 1.0   : hover at x = 0
      - 1 <= t < 4: smoothly move from x = 0 to x = 0.5
      - t >= 4.0  : hover at x = 0.5
    Altitude is kept at z_ref = 0.5 the whole time.
    """
    x_start = 0.0
    x_final = 0.5

    if t < 1.0:
        x = x_start
        xdot = 0.0
    elif t < 4.0:
        # smooth S-curve using cosine interpolation
        tau = (t - 1.0) / (4.0 - 1.0)  # in [0, 1]
        s = 0.5 * (1 - np.cos(np.pi * tau))        # position blend
        s_dot = 0.5 * np.pi * np.sin(np.pi * tau) / (4.0 - 1.0)  # derivative
        x = x_start + s * (x_final - x_start)
        xdot = s_dot * (x_final - x_start)
    else:
        x = x_final
        xdot = 0.0

    z = 0.5
    zdot = 0.0
    theta = 0.0
    thetadot = 0.0

    return np.array([x, xdot, z, zdot, theta, thetadot])


def main():
    """
    Main function for LQR smooth trajectory tracking simulation.

    This function:
        1. Defines system parameters (same as main_lqr.py)
        2. Designs LQR controller
        3. Defines smooth reference trajectory function
        4. Implements LQR tracking control law
        5. Runs simulation from perturbed initial condition
        6. Plots results showing tracking performance
        7. Creates animation
    """
    # System parameters (same as main_lqr.py)
    m = 1.0
    I = 0.02
    l = 0.25
    g = 9.81

    params = {"m": m, "I": I, "l": l, "g": g}

    print("=" * 60)
    print("LQR SMOOTH TRAJECTORY TRACKING SIMULATION")
    print("=" * 60)

    # Linearization around hover
    print("\nLinearizing system around hover equilibrium...")
    A, B = linearize_quadrotor(m=m, I=I, l=l, g=g)
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # Use the same Q, R structure as in main_lqr.py
    Q = np.diag([
        1.0,   # x
        0.1,   # xdot
        10.0,  # z
        5.0,   # zdot
        100.0, # theta
        10.0   # thetadot
    ])

    R = np.diag([
        1.0,  # FT
        1.0   # tau
    ])

    print("\nDesigning LQR controller...")
    K, P, eigvals = control.lqr(A, B, Q, R)
    print(f"LQR gain matrix K shape: {K.shape}")
    print(f"Closed-loop eigenvalues: {eigvals}")

    # Hover equilibrium
    z_star = 0.5
    x_hover = 0.0
    x_ref_eq = np.array([x_hover, 0.0, z_star, 0.0, 0.0, 0.0])
    u_hover = np.array([m * g, 0.0])

    print(f"\nHover equilibrium state: {x_ref_eq}")
    print(f"Hover control input: {u_hover}")

    # LQR tracking law
    def u_func(t, state):
        """
        LQR tracking control law: u = u_hover - K @ (state - x_ref)
        
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
        x_ref = reference_trajectory(t)
        x_tilde = state - x_ref
        u_tilde = -K @ x_tilde
        return u_hover + u_tilde

    # Simulation setup
    t_span = (0.0, 8.0)
    dt = 0.01

    # Start slightly perturbed
    state0 = np.array([0.0, 0.0, 0.8, 0.0, 0.05, 0.0])

    print(f"\nInitial state: x0 = {state0}")
    print(f"Reference at t=0: x_ref(0) = {reference_trajectory(0.0)}")
    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(state0, u_func, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"Final reference: x_ref({t[-1]:.2f}) = {reference_trajectory(t[-1])}")

    # Compute reference trajectory and errors
    x_ref_traj = np.array([reference_trajectory(ti) for ti in t])
    errors = states - x_ref_traj

    # Get project root directory (parent of code directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    # Plot results
    print("\nGenerating plots...")

    # ====================================================================
    # Plot 1: States vs reference
    # ====================================================================
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8))
    
    # x vs x_ref
    axes1[0].plot(t, states[:, 0], 'b-', linewidth=2, label='x(t)')
    axes1[0].plot(t, x_ref_traj[:, 0], 'r--', linewidth=2, label='x_ref(t)')
    axes1[0].set_ylabel('x (m)', fontsize=12)
    axes1[0].set_title('Horizontal Position')
    axes1[0].legend(fontsize=11)
    axes1[0].grid(True, alpha=0.3)
    
    # z vs z_ref
    axes1[1].plot(t, states[:, 2], 'b-', linewidth=2, label='z(t)')
    axes1[1].plot(t, x_ref_traj[:, 2], 'r--', linewidth=2, label='z_ref(t)')
    axes1[1].set_ylabel('z (m)', fontsize=12)
    axes1[1].set_title('Vertical Position')
    axes1[1].legend(fontsize=11)
    axes1[1].grid(True, alpha=0.3)
    
    # theta vs theta_ref
    axes1[2].plot(t, states[:, 4], 'b-', linewidth=2, label='theta(t)')
    axes1[2].plot(t, x_ref_traj[:, 4], 'r--', linewidth=2, label='theta_ref(t)')
    axes1[2].set_xlabel('Time (s)', fontsize=12)
    axes1[2].set_ylabel('theta (rad)', fontsize=12)
    axes1[2].set_title('Pitch Angle')
    axes1[2].legend(fontsize=11)
    axes1[2].grid(True, alpha=0.3)
    
    plt.suptitle('LQR Smooth Trajectory Tracking - States vs Reference', fontsize=14)
    plt.tight_layout()
    
    states_path = os.path.join(visuals_dir, "tracking_smooth_states.png")
    plt.savefig(states_path, dpi=150, bbox_inches="tight")
    print(f"Saved states plot to {states_path}")

    # ====================================================================
    # Plot 2: Tracking errors
    # ====================================================================
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Error in x
    axes2[0].plot(t, errors[:, 0], 'b-', linewidth=2)
    axes2[0].set_ylabel('e_x [m]', fontsize=12)
    axes2[0].set_title('Tracking Error in x')
    axes2[0].grid(True, alpha=0.3)
    
    # Error in z
    axes2[1].plot(t, errors[:, 2], 'b-', linewidth=2)
    axes2[1].set_ylabel('e_z [m]', fontsize=12)
    axes2[1].set_title('Tracking Error in z')
    axes2[1].grid(True, alpha=0.3)
    
    # Error in theta
    axes2[2].plot(t, errors[:, 4], 'b-', linewidth=2)
    axes2[2].set_xlabel('Time [s]', fontsize=12)
    axes2[2].set_ylabel('e_theta [rad]', fontsize=12)
    axes2[2].set_title('Tracking Error in theta')
    axes2[2].grid(True, alpha=0.3)
    
    plt.suptitle('LQR Smooth Trajectory Tracking - Errors', fontsize=14)
    plt.tight_layout()
    
    errors_path = os.path.join(visuals_dir, "tracking_smooth_errors.png")
    plt.savefig(errors_path, dpi=150, bbox_inches="tight")
    print(f"Saved errors plot to {errors_path}")

    # ====================================================================
    # Plot 3: Control inputs
    # ====================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    
    # FT vs t
    axes3[0].plot(t, inputs[:, 0], 'r-', linewidth=1.5)
    axes3[0].set_xlabel('Time (s)', fontsize=12)
    axes3[0].set_ylabel('FT (N)', fontsize=12)
    axes3[0].set_title('Total Thrust vs Time')
    axes3[0].grid(True, alpha=0.3)
    
    # tau vs t
    axes3[1].plot(t, inputs[:, 1], 'g-', linewidth=1.5)
    axes3[1].set_xlabel('Time (s)', fontsize=12)
    axes3[1].set_ylabel('tau (N⋅m)', fontsize=12)
    axes3[1].set_title('Torque vs Time')
    axes3[1].grid(True, alpha=0.3)
    
    plt.suptitle('LQR Smooth Trajectory Tracking - Control Inputs', fontsize=14)
    plt.tight_layout()
    
    inputs_path = os.path.join(visuals_dir, "tracking_smooth_inputs.png")
    plt.savefig(inputs_path, dpi=150, bbox_inches="tight")
    print(f"Saved inputs plot to {inputs_path}")

    # ====================================================================
    # Animation
    # ====================================================================
    print("\nCreating animation...")
    anim_path = os.path.join(visuals_dir, "lqr_tracking_smooth.mp4")
    anim = animate_quadrotor(t, states, params, ref_states=x_ref_traj, save_path=anim_path, fps=30)
    print(f"Animation saved to {anim_path}")

    # Show plots and animation
    # Note: The animation object must be stored in a variable to prevent garbage collection
    plt.show()

    print("\n" + "=" * 60)
    print("LQR SMOOTH TRAJECTORY TRACKING SIMULATION")
    print("=" * 60)
    print(f"Number of time steps: {len(t)}")
    print("Saved figures and animation to:", visuals_dir)
    print("\nThe plots show that the LQR controller successfully tracks")
    print("the smooth reference trajectory, smoothly transitioning from")
    print("x=0 to x=0.5 while maintaining z=0.5 and theta=0.")


if __name__ == "__main__":
    main()

