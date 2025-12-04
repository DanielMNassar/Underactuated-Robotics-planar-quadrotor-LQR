"""
LQR Controlled Simulation Main Script

This script runs a closed-loop simulation of the planar quadrotor
using LQR control for stabilization.

Usage:
    python main_lqr.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode
from code.controllers import design_lqr_controller
from code.visualization import plot_states, plot_inputs, animate_quadrotor


def main():
    """
    Main function for LQR-controlled simulation.

    This function:
        1. Defines system parameters
        2. Designs LQR controller
        3. Sets equilibrium state and initial conditions
        4. Runs controlled simulation
        5. Plots results showing stabilization
    """
    # System parameters
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
    print("LQR CONTROLLED SIMULATION - PLANAR QUADROTOR")
    print("=" * 60)

    # Design LQR controller
    print("\nDesigning LQR controller...")
    K, A, B = design_lqr_controller(m=m, I=I, l=l, g=g)
    print(f"LQR gain matrix K shape: {K.shape}")
    print(f"K = \n{K}")

    # Define equilibrium state: x* = [0, 0, z*, 0, 0, 0]
    z_star = 0.5  # Desired altitude
    state_ref = np.array([0.0, 0.0, z_star, 0.0, 0.0, 0.0])
    u_ref = np.array([m * g, 0.0])  # Hover thrust

    print(f"\nEquilibrium state: x* = {state_ref}")
    print(f"Equilibrium input: u* = {u_ref}")

    # Initial conditions (perturbed from equilibrium)
    # Small nonzero theta, x, z
    state0 = np.array([0.2, 0.0, 0.8, 0.0, 0.15, 0.0])  # Perturbed initial condition

    print(f"\nInitial state: x0 = {state0}")
    print(f"Initial error: x0 - x* = {state0 - state_ref}")

    # Define control law: u(t, x) = u* - K @ (x - x*)
    def u_controller(t, state):
        """
        LQR control law: u = u* - K @ (x - x*)
        
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
        error = state - state_ref
        u = u_ref - K @ error
        return u

    # Time span: 5-10 seconds
    t_span = (0.0, 8.0)
    dt = 0.01

    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(state0, u_controller, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"Final error: x_final - x* = {states[-1] - state_ref}")

    # Plot results
    print("\nGenerating plots...")

    # Get project root directory (parent of code directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # Plot states: x(t), z(t), theta(t) (and optionally others)
    fig_states, axes_states = plot_states(t, states)
    plt.suptitle('LQR Controlled Planar Quadrotor - State Trajectories', fontsize=14)
    
    # Save state plot
    states_path = os.path.join(visuals_dir, 'lqr_states.png')
    plt.savefig(states_path, dpi=150, bbox_inches='tight')
    print(f"Saved state plot to {states_path}")

    # Plot control inputs
    fig_inputs, axes_inputs = plot_inputs(t, inputs)
    plt.suptitle('LQR Controlled Planar Quadrotor - Control Inputs', fontsize=14)
    
    # Save input plot
    inputs_path = os.path.join(visuals_dir, 'lqr_inputs.png')
    plt.savefig(inputs_path, dpi=150, bbox_inches='tight')
    print(f"Saved input plot to {inputs_path}")

    # Create focused plot showing x(t), z(t), theta(t) together
    fig_focused, axes_focused = plt.subplots(3, 1, figsize=(10, 8))
    
    # x(t)
    axes_focused[0].plot(t, states[:, 0], 'b-', linewidth=2, label='x(t)')
    axes_focused[0].axhline(y=state_ref[0], color='r', linestyle='--', linewidth=1.5, label='Reference')
    axes_focused[0].set_ylabel('x (m)')
    axes_focused[0].set_title('Horizontal Position')
    axes_focused[0].legend()
    axes_focused[0].grid(True, alpha=0.3)
    
    # z(t)
    axes_focused[1].plot(t, states[:, 2], 'b-', linewidth=2, label='z(t)')
    axes_focused[1].axhline(y=state_ref[2], color='r', linestyle='--', linewidth=1.5, label='Reference')
    axes_focused[1].set_ylabel('z (m)')
    axes_focused[1].set_title('Vertical Position')
    axes_focused[1].legend()
    axes_focused[1].grid(True, alpha=0.3)
    
    # theta(t)
    axes_focused[2].plot(t, states[:, 4], 'b-', linewidth=2, label='theta(t)')
    axes_focused[2].axhline(y=state_ref[4], color='r', linestyle='--', linewidth=1.5, label='Reference')
    axes_focused[2].set_xlabel('Time (s)')
    axes_focused[2].set_ylabel('theta (rad)')
    axes_focused[2].set_title('Pitch Angle')
    axes_focused[2].legend()
    axes_focused[2].grid(True, alpha=0.3)
    
    plt.suptitle('LQR Controller Stabilization - Key States', fontsize=14)
    plt.tight_layout()
    stabilization_path = os.path.join(visuals_dir, 'lqr_stabilization.png')
    plt.savefig(stabilization_path, dpi=150, bbox_inches='tight')
    print(f"Saved focused plot to {stabilization_path}")

    # Create animation
    print("\nCreating animation...")
    animation_path = os.path.join(visuals_dir, 'lqr_quadrotor.mp4')
    anim = animate_quadrotor(t, states, params, save_path=animation_path, fps=30)
    print(f"Animation saved to {animation_path}")

    # Show plots and animation
    # Note: The animation object must be stored in a variable to prevent garbage collection
    plt.show()

    print("\n" + "=" * 60)
    print("LQR simulation complete!")
    print("=" * 60)
    print("\nThe plots show that the LQR controller successfully stabilizes")
    print("the system back to the equilibrium state from the perturbed")
    print("initial condition.")


if __name__ == "__main__":
    main()

