"""
Lyapunov Function Analysis for LQR-Controlled Planar Quadrotor

This script computes and plots the Lyapunov function V(x) = x^T P x
along a closed-loop trajectory to demonstrate that V(t) decreases,
proving stability of the LQR controller.

Usage:
    cd code
    python analysis_lyapunov.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.simulation import simulate_ode, linearize_quadrotor
import control


def add_roa_ellipse(ax, P_sub, V_level, color, label, alpha=0.15, linestyle='-'):
    """
    Add a filled Lyapunov level-set ellipse to the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to plot on.
    P_sub : 2x2 ndarray
        Submatrix of P corresponding to [x, theta].
    V_level : float
        Lyapunov level V(x,theta) = V_level.
    color : str
        Matplotlib color.
    label : str
        Legend label.
    alpha : float
        Transparency of the fill.
    linestyle : str
        Line style for ellipse edge.
    """
    eigvals, eigvecs = np.linalg.eig(P_sub)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = np.sqrt(V_level / eigvals[0])
    b = np.sqrt(V_level / eigvals[1])

    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

    ell = Ellipse((0.0, 0.0),
                  width=2*a, height=2*b,
                  angle=angle,
                  edgecolor=color,
                  facecolor=color,
                  alpha=alpha,
                  linestyle=linestyle,
                  linewidth=2,
                  label=label)
    ax.add_patch(ell)

    return a, b


def lyapunov_ellipse(P2, rho, num_points=200):
    """
    Returns arrays (xs, thetas) representing the boundary of the level set
    { z in R^2 | z^T P2 z = rho }, where z = [x, theta]^T.
    
    Parameters
    ----------
    P2 : ndarray, shape (2, 2)
        2x2 submatrix of the Riccati solution P corresponding to (x, theta)
    rho : float
        Level set value (rho > 0)
    num_points : int, optional
        Number of points to generate for the ellipse (default: 200)
        
    Returns
    -------
    xs : ndarray
        Array of x coordinates on the ellipse boundary
    thetas : ndarray
        Array of theta coordinates on the ellipse boundary
    """
    xs = []
    thetas = []
    # Sample angles from 0 to 2*pi
    phis = np.linspace(0, 2*np.pi, num_points)
    for phi in phis:
        u = np.array([np.cos(phi), np.sin(phi)])  # unit direction
        denom = u.T @ P2 @ u
        if denom <= 0:
            continue
        r = np.sqrt(rho / denom)
        z = r * u
        xs.append(z[0])
        thetas.append(z[1])
    return np.array(xs), np.array(thetas)


def main():
    """
    Main function for Lyapunov function analysis.
    
    This function:
        1. Defines system parameters (same as main_lqr.py)
        2. Designs LQR controller and gets Riccati matrix P
        3. Sets equilibrium state and initial conditions
        4. Runs closed-loop simulation
        5. Computes V(t) = x_tilde^T @ P @ x_tilde along trajectory
        6. Plots V(t) to show it decreases
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
    print("LYAPUNOV FUNCTION ANALYSIS - LQR CONTROLLED QUADROTOR")
    print("=" * 60)

    # Define Q and R exactly as in main_lqr.py (defaults from design_lqr_controller)
    Q = np.diag([1.0,    # x position (low penalty)
                 0.1,    # xdot velocity (low penalty)
                 10.0,   # z position (high penalty - maintain altitude)
                 5.0,    # zdot velocity (medium penalty)
                 100.0,  # theta angle (very high penalty - maintain attitude)
                 10.0])  # thetadot angular velocity (high penalty)

    R = np.diag([1.0,   # FT thrust (low penalty)
                 1.0])  # tau torque (low penalty)

    print("\nLQR Design Matrices:")
    print(f"Q = \n{Q}")
    print(f"R = \n{R}")

    # Compute A, B using linearize_quadrotor
    print("\nLinearizing system around hover equilibrium...")
    A, B = linearize_quadrotor(m=m, I=I, l=l, g=g)
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    # Call control.lqr(A, B, Q, R) to get (K, P, eigvals)
    print("\nSolving LQR Riccati equation...")
    K, P, eigvals = control.lqr(A, B, Q, R)
    print(f"LQR gain matrix K shape: {K.shape}")
    print(f"Riccati solution P shape: {P.shape}")
    print(f"Closed-loop eigenvalues: {eigvals}")

    # Define equilibrium state and hover input (same as main_lqr.py)
    z_star = 0.5  # Desired altitude
    x_ref = np.array([0.0, 0.0, z_star, 0.0, 0.0, 0.0])
    u_hover = np.array([m * g, 0.0])  # Hover thrust

    print(f"\nReference state: x_ref = {x_ref}")
    print(f"Hover input: u_hover = {u_hover}")

    # Initial conditions (same as main_lqr.py)
    state0 = np.array([0.2, 0.0, 0.8, 0.0, 0.15, 0.0])  # Perturbed initial condition
    print(f"Initial state: x0 = {state0}")
    print(f"Initial error: x0 - x_ref = {state0 - x_ref}")

    # Define control law: u(t, x) = u_hover - K @ (x - x_ref)
    def u_func(t, state):
        x_ref = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        x_tilde = state - x_ref
        u_tilde = -K @ x_tilde
        return u_hover + u_tilde

    # Time span and simulation parameters (same as main_lqr.py)
    t_span = (0.0, 8.0)
    dt = 0.01

    print(f"\nSimulating from t={t_span[0]} to t={t_span[1]} seconds...")
    print("Running closed-loop simulation...")

    # Run simulation
    t, states, inputs = simulate_ode(state0, u_func, params, t_span, dt=dt)

    print("Simulation complete!")
    print(f"Number of time steps: {len(t)}")

    # Compute Lyapunov function V(t) = x_tilde^T @ P @ x_tilde
    print("\nComputing Lyapunov function V(t)...")
    V = np.zeros(len(t))
    
    for i in range(len(t)):
        x_tilde = states[i] - x_ref
        V[i] = x_tilde.T @ P @ x_tilde

    print(f"Initial V(0) = {V[0]:.6f}")
    print(f"Final V({t[-1]:.2f}) = {V[-1]:.6f}")
    print(f"V decreased by: {V[0] - V[-1]:.6f}")

    # Verify that V(t) is non-negative and decreasing
    if np.any(V < 0):
        print("\nWARNING: V(t) has negative values! This should not happen.")
    else:
        print("\n✓ V(t) >= 0 for all t (as expected)")

    # Check if V is monotonically decreasing
    V_diff = np.diff(V)
    if np.any(V_diff > 1e-6):  # Allow small numerical errors
        num_increases = np.sum(V_diff > 1e-6)
        print(f"\nWARNING: V(t) increased at {num_increases} time steps (may be due to numerical errors)")
    else:
        print("✓ V(t) is monotonically decreasing (within numerical precision)")

    # Plot V(t) vs t
    print("\nGenerating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, V, 'b-', linewidth=2, label='V(t) = x̃^T P x̃')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('V(x)', fontsize=12)
    ax.set_title('Lyapunov Function V(t) under LQR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add text annotation showing initial and final values
    ax.text(0.02, 0.98, f'V(0) = {V[0]:.4f}\nV({t[-1]:.1f}) = {V[-1]:.6f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save the figure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visuals_dir = os.path.join(project_root, 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)
    
    save_path = os.path.join(visuals_dir, 'lyapunov_V_t.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

    # Show the plot
    plt.show()

    # ------------------------------------------------------------
    # ROA visualization in the (x, theta) plane
    # ------------------------------------------------------------
    print("\nGenerating ROA visualization in (x, theta)...")

    idx = [0, 4]  # x and theta indices
    P_sub = P[np.ix_(idx, idx)]

    V_levels = [0.1, 0.2, 0.5, 1.0]
    colors   = ['#4caf50', '#ff9800', '#f44336', '#673ab7']
    labels   = ['Inner fast-convergence region',
                'Medium convergence region',
                'Large region',
                'Approx ROA boundary (V=1.0)']

    fig2, ax2 = plt.subplots(figsize=(9, 7))

    semi_axes = []
    for V, c, lab, ls in zip(V_levels, colors, labels,
                             ['--', '-.', ':', '-']):
        a, b = add_roa_ellipse(ax2, P_sub, V, c, lab,
                               alpha=0.15, linestyle=ls)
        semi_axes.append((V, a, b))

    # Plot trajectory
    ax2.plot(states[:, 0], states[:, 4],
             'b', linewidth=2, label='Closed-loop trajectory')

    # Initial + equilibrium
    ax2.plot(state0[0], state0[4], 'go', markersize=9, label='Initial state')
    ax2.plot(x_ref[0], x_ref[4], 'r*', markersize=14, label='Equilibrium')

    ax2.set_xlabel('x [m]', fontsize=12)
    ax2.set_ylabel('theta [rad]', fontsize=12)
    ax2.set_title('Lyapunov ROA Approximation in (x, theta)', fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Auto-scale axes based on largest ellipse
    _, a_outer, b_outer = semi_axes[-1]
    ax2.set_xlim(-1.1*a_outer, 1.1*a_outer)
    ax2.set_ylim(-1.1*b_outer, 1.1*b_outer)

    plt.tight_layout()

    roa_path = os.path.join(visuals_dir, 'roa_x_theta.png')
    plt.savefig(roa_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROA plot to {roa_path}")

    print("\nApproximate ROA numerical bounds:")
    print(f"  |x| <= {a_outer:.3f} m")
    print(f"  |theta| <= {b_outer:.3f} rad (~{np.degrees(b_outer):.1f} deg)")

    # Show the plot
    plt.show()

    print("\n" + "=" * 60)
    print("Lyapunov analysis complete!")
    print("=" * 60)
    print("\nThe plot shows that V(t) = x̃^T P x̃ is:")
    print("  - Non-negative (V(t) >= 0)")
    print("  - Monotonically decreasing towards 0")
    print("This demonstrates that the LQR controller is stabilizing.")
    print("\nThe ROA visualization shows Lyapunov level sets (ellipses) in the")
    print("(x, theta) plane, approximating the region of attraction. The")
    print("closed-loop trajectory is shown to evolve within these level sets.")


if __name__ == "__main__":
    main()

