"""
Open-Loop Simulation Main Script

This script runs an open-loop simulation of the planar quadrotor
with constant hover control inputs (no feedback control).

The simulation demonstrates that the system is unstable in open loop,
as a small initial perturbation causes the system to diverge.

Usage:
    cd code
    python main_openloop.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode


def main():
    """
    Main function for open-loop simulation.

    This function:
        1. Defines system parameters
        2. Sets initial conditions with small perturbation
        3. Defines constant open-loop control (hover thrust, zero torque)
        4. Runs simulation
        5. Plots results showing system instability
    """
    # System parameters
    params = {
        "m": 1.0,
        "I": 0.02,
        "l": 0.25,
        "g": 9.81,
    }

    # Initial state with small perturbation
    # state = [x, xdot, z, zdot, theta, thetadot]
    x0 = np.array([
        0.0,              # x
        0.0,              # xdot
        0.5,              # z (0.5 m height)
        0.0,              # zdot
        np.deg2rad(10.0), # theta = 10 degrees
        0.0               # thetadot
    ])

    # Constant open-loop control: hover thrust and zero torque
    u_hover = np.array([params["m"] * params["g"], 0.0])

    def u_func(t, state):
        """
        Open-loop control function (no feedback).
        
        Parameters
        ----------
        t : float
            Current time
        state : array_like, shape (6,)
            Current state vector (not used in open-loop)
            
        Returns
        -------
        u : ndarray, shape (2,)
            Constant control input [FT, tau]
        """
        # No feedback, just hover thrust and zero torque
        return u_hover

    # Time span for simulation
    t_span = (0.0, 5.0)

    print("=" * 60)
    print("OPEN-LOOP SIMULATION - PLANAR QUADROTOR")
    print("=" * 60)
    print(f"\nSystem parameters:")
    print(f"  m = {params['m']} kg")
    print(f"  I = {params['I']} kg*m^2")
    print(f"  l = {params['l']} m")
    print(f"  g = {params['g']} m/s^2")
    print(f"\nInitial state:")
    print(f"  x0 = {x0}")
    print(f"  Initial pitch angle: {np.rad2deg(x0[4]):.1f} degrees")
    print(f"\nControl input (constant):")
    print(f"  FT = {u_hover[0]:.2f} N (hover thrust)")
    print(f"  tau = {u_hover[1]:.2f} N*m (zero torque)")
    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(
        state0=x0,
        u_func=u_func,
        params=params,
        t_span=t_span,
        dt=0.01,
        method="RK45"
    )

    print("Simulation complete!")
    print(f"Final state: x_final = {states[-1]}")
    print(f"\nNote: The system diverges due to the initial pitch perturbation")
    print("      and lack of feedback control. This demonstrates instability.")

    # Create plots
    print("\nGenerating plots...")

    # Get project root directory (parent of code directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot x(t)
    axes[0].plot(t, states[:, 0], 'b-', linewidth=2)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('x (m)', fontsize=12)
    axes[0].set_title('Open-loop x(t)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot z(t)
    axes[1].plot(t, states[:, 2], 'b-', linewidth=2)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('z (m)', fontsize=12)
    axes[1].set_title('Open-loop z(t)', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Plot theta(t)
    axes[2].plot(t, states[:, 4], 'b-', linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('theta (rad)', fontsize=12)
    axes[2].set_title('Open-loop theta(t)', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Open-Loop Simulation - System Instability', fontsize=16)
    plt.tight_layout()

    # Save figure
    openloop_path = os.path.join(visuals_dir, 'openloop_states.png')
    plt.savefig(openloop_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {openloop_path}")

    # Show plots
    plt.show()

    print("\n" + "=" * 60)
    print("Open-loop simulation complete!")
    print("=" * 60)
    print("\nThe plots clearly show that the system diverges or behaves")
    print("unstably without feedback control, demonstrating the need")
    print("for closed-loop control (e.g., LQR) to stabilize the system.")


if __name__ == "__main__":
    main()
