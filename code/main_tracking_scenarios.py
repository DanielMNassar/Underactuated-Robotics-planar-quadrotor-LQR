"""
LQR Trajectory Tracking Scenarios for Planar Quadrotor

This script demonstrates LQR control for tracking different types of reference
trajectories. Change the SCENARIO constant at the top to select different
trajectory types.

Usage:
    cd code
    python main_tracking_scenarios.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Allow running from inside code/ with `python main_tracking_scenarios.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode, linearize_quadrotor
from code.visualization import animate_quadrotor
from code.controllers import design_lqr_controller
import control  # python-control

# ====================================================================
# SCENARIO SELECTION
# ====================================================================
SCENARIO = 1  # 0 = smooth, 1 = sine, 2 = ramp, 3 = S-curve, 4 = square wave

# Scenario names for display and filenames
SCENARIO_NAMES = {
    0: "smooth",
    1: "sine",
    2: "ramp",
    3: "s_curve",
    4: "square"
}


def reference_trajectory(t, scenario):
    """
    Return reference state x_ref(t) for the given scenario.
    
    State format: [x_ref, xdot_ref, z_ref, zdot_ref, theta_ref, thetadot_ref]
    
    Parameters
    ----------
    t : float
        Current time
    scenario : int
        Scenario number (0-4)
        
    Returns
    -------
    x_ref : ndarray, shape (6,)
        Reference state vector
    """
    z_ref = 0.5
    zdot_ref = 0.0
    theta_ref = 0.0
    thetadot_ref = 0.0
    
    if scenario == 0:
        # Smooth transition (same as main_tracking_smooth.py)
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
            
    elif scenario == 1:
        # Sinusoidal
        A = 0.5   # amplitude [m]
        omega = 0.8  # rad/s
        x = A * np.sin(omega * t)
        xdot = A * omega * np.cos(omega * t)
        
    elif scenario == 2:
        # Ramp (saturated)
        v = 0.2   # m/s
        x = np.minimum(v * t, 1.0)
        if v * t < 1.0:
            xdot = v
        else:
            xdot = 0.0
            
    elif scenario == 3:
        # S-curve / logistic
        x_final = 0.5
        alpha = 1.5
        t0 = 3.0
        x = x_final / (1.0 + np.exp(-alpha * (t - t0)))
        # Derivative of logistic function
        exp_term = np.exp(-alpha * (t - t0))
        xdot = x_final * alpha * exp_term / ((1.0 + exp_term) ** 2)
        
    elif scenario == 4:
        # Square wave
        if t < 2.0:
            x = 0.5
            xdot = 0.0
        elif t < 4.0:
            x = -0.5
            xdot = 0.0
        else:
            x = 0.5
            xdot = 0.0
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return np.array([x, xdot, z_ref, zdot_ref, theta_ref, thetadot_ref])


def main():
    """
    Main function for LQR trajectory tracking with different scenarios.
    """
    # Get scenario name
    if SCENARIO not in SCENARIO_NAMES:
        raise ValueError(f"Invalid scenario: {SCENARIO}. Must be 0-4.")
    
    scenario_name = SCENARIO_NAMES[SCENARIO]
    print("=" * 60)
    print(f"Running LQR tracking scenario {SCENARIO} - {scenario_name}")
    print("=" * 60)
    
    # System parameters (same as main_lqr.py)
    m = 1.0
    I = 0.02
    l = 0.25
    g = 9.81

    params = {"m": m, "I": I, "l": l, "g": g}

    # Design LQR controller (reuse from controllers.py)
    print("\nDesigning LQR controller...")
    K, A, B = design_lqr_controller(m=m, I=I, l=l, g=g)
    print(f"LQR gain matrix K shape: {K.shape}")

    # Hover equilibrium
    z_star = 0.5
    u_hover = np.array([m * g, 0.0])

    print(f"Hover control input: {u_hover}")

    # Define control law
    def u_func(t, state):
        """
        LQR tracking control law: u = u_hover - K @ (state - x_ref)
        """
        x_ref = reference_trajectory(t, SCENARIO)
        x_tilde = state - x_ref
        u_tilde = -K @ x_tilde
        return u_hover + u_tilde

    # Simulation setup
    t_span = (0.0, 8.0)
    dt = 0.01

    # Start slightly perturbed
    state0 = np.array([0.0, 0.0, 0.8, 0.0, 0.05, 0.0])

    print(f"\nInitial state: x0 = {state0}")
    print(f"Reference at t=0: x_ref(0) = {reference_trajectory(0.0, SCENARIO)}")
    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(state0, u_func, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"Final reference: x_ref({t[-1]:.2f}) = {reference_trajectory(t[-1], SCENARIO)}")

    # Compute reference trajectory and errors
    x_ref_traj = np.array([reference_trajectory(ti, SCENARIO) for ti in t])
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
    axes1[0].set_title(f'Horizontal Position - Scenario {SCENARIO} ({scenario_name})')
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
    
    plt.suptitle(f'LQR Tracking - Scenario {SCENARIO} ({scenario_name}) - States vs Reference', fontsize=14)
    plt.tight_layout()
    
    states_path = os.path.join(visuals_dir, f"tracking_scenario{SCENARIO}_{scenario_name}_states.png")
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
    
    plt.suptitle(f'LQR Tracking - Scenario {SCENARIO} ({scenario_name}) - Errors', fontsize=14)
    plt.tight_layout()
    
    errors_path = os.path.join(visuals_dir, f"tracking_scenario{SCENARIO}_{scenario_name}_errors.png")
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
    
    plt.suptitle(f'LQR Tracking - Scenario {SCENARIO} ({scenario_name}) - Control Inputs', fontsize=14)
    plt.tight_layout()
    
    inputs_path = os.path.join(visuals_dir, f"tracking_scenario{SCENARIO}_{scenario_name}_inputs.png")
    plt.savefig(inputs_path, dpi=150, bbox_inches="tight")
    print(f"Saved inputs plot to {inputs_path}")

    # ====================================================================
    # Animation
    # ====================================================================
    print("\nCreating animation...")
    anim_path = os.path.join(visuals_dir, f"tracking_scenario{SCENARIO}_{scenario_name}.mp4")
    anim = animate_quadrotor(t, states, params, ref_states=x_ref_traj, save_path=anim_path, fps=30)
    print(f"Animation saved to {anim_path}")

    # Show plots and animation
    # Note: The animation object must be stored in a variable to prevent garbage collection
    plt.show()

    print("\n" + "=" * 60)
    print(f"LQR Tracking Scenario {SCENARIO} ({scenario_name}) complete!")
    print("=" * 60)
    print(f"Number of time steps: {len(t)}")
    print("Saved figures and animation to:", visuals_dir)
    print("\nTo run a different scenario, change SCENARIO at the top of this file.")


if __name__ == "__main__":
    main()

