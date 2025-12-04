"""
Visualization Module for Planar Quadrotor

This module provides plotting and animation utilities for visualizing
the planar quadrotor simulation results.

Functions:
    plot_states: Plot state trajectories over time
    plot_inputs: Plot control input trajectories over time
    animate_quadrotor: Create animation of quadrotor motion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_states(t, states, labels=None, figsize=(12, 8)):
    """
    Plot state trajectories over time.

    Parameters
    ----------
    t : array_like
        Time array
    states : ndarray, shape (N, 6)
        State trajectory [x, xdot, z, zdot, theta, thetadot]
    labels : list of str, optional
        Custom labels for subplots
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : ndarray of matplotlib.axes.Axes
        Array of axes objects

    Notes
    -----
    Creates 6 subplots for each state variable.
    """
    states = np.array(states)
    t = np.array(t)

    # Default labels
    if labels is None:
        labels = ['x (m)', 'xdot (m/s)', 'z (m)', 'zdot (m/s)', 
                  'theta (rad)', 'thetadot (rad/s)']

    # State names for titles
    state_names = ['x', 'xdot', 'z', 'zdot', 'theta', 'thetadot']

    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()

    # Plot each state
    for i in range(6):
        axes[i].plot(t, states[:, i], 'b-', linewidth=1.5)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(labels[i])
        axes[i].set_title(f'{state_names[i]} vs Time')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_inputs(t, inputs, figsize=(10, 4)):
    """
    Plot control input trajectories over time.

    Parameters
    ----------
    t : array_like
        Time array
    inputs : ndarray, shape (N, 2)
        Control input trajectory [FT, tau]
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : ndarray of matplotlib.axes.Axes
        Array of axes objects

    Notes
    -----
    Creates 2 subplots for thrust and torque.
    """
    inputs = np.array(inputs)
    t = np.array(t)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot thrust
    axes[0].plot(t, inputs[:, 0], 'r-', linewidth=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('FT (N)')
    axes[0].set_title('Total Thrust vs Time')
    axes[0].grid(True, alpha=0.3)

    # Plot torque
    axes[1].plot(t, inputs[:, 1], 'g-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('tau (N⋅m)')
    axes[1].set_title('Torque vs Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def animate_quadrotor(t, states, params, ref_states=None, save_path=None, fps=30):
    """
    Create an animation of the planar quadrotor motion in the x–z plane.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time array.
    states : ndarray, shape (N, 6)
        State trajectory [x, xdot, z, zdot, theta, thetadot].
    params : dict
        System parameters. Must contain at least 'l' (half arm length).
    ref_states : ndarray, shape (N, 6), optional
        Reference trajectory [x_ref, xdot_ref, z_ref, zdot_ref, theta_ref, thetadot_ref].
        If provided, shows reference point and path in the animation.
    save_path : str or None
        If not None, path to save the animation (mp4 or gif).
    fps : int
        Frames per second for saving.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object

    Notes
    -----
    The animation shows:
        - Quadrotor body as a line between the two rotors
        - Two rotors as small circles at the ends of the body
        - Fixed x-z axes limits for all frames
        - Trajectory path (optional, can be added)
        - If ref_states is provided: reference point (red star) and reference path (red dashed)
    """
    # Convert inputs to numpy arrays
    t = np.array(t)
    states = np.array(states)
    
    # Handle reference states if provided
    if ref_states is not None:
        ref_states = np.array(ref_states)
        x_ref = ref_states[:, 0]  # Reference horizontal position
        z_ref = ref_states[:, 2]  # Reference vertical position
    else:
        x_ref = None
        z_ref = None
    
    # Extract parameters
    l = params['l']  # Half-distance between rotors
    
    # Extract state variables
    x = states[:, 0]  # Horizontal position
    z = states[:, 2]  # Vertical position
    theta = states[:, 4]  # Pitch angle
    
    # Compute rotor positions for each time step
    # Rotor 1 (left): position relative to center, rotated by theta
    # Rotor 2 (right): position relative to center, rotated by theta
    # For a bar of length 2l centered at (x, z) rotated by theta:
    # - Left end: (x - l*cos(theta), z - l*sin(theta))
    # - Right end: (x + l*cos(theta), z + l*sin(theta))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotor1_x = x - l * cos_theta
    rotor1_z = z - l * sin_theta
    rotor2_x = x + l * cos_theta
    rotor2_z = z + l * sin_theta
    
    # Set fixed axis limits based on the entire trajectory
    # Include reference states if provided
    x_min = np.min([np.min(rotor1_x), np.min(rotor2_x)])
    x_max = np.max([np.max(rotor1_x), np.max(rotor2_x)])
    z_min = np.min([np.min(rotor1_z), np.min(rotor2_z)])
    z_max = np.max([np.max(rotor1_z), np.max(rotor2_z)])
    
    # Include reference trajectory in limits if provided
    if ref_states is not None:
        x_min = min(x_min, np.min(x_ref))
        x_max = max(x_max, np.max(x_ref))
        z_min = min(z_min, np.min(z_ref))
        z_max = max(z_max, np.max(z_ref))
    
    # Add padding
    x_min -= 0.5
    x_max += 0.5
    z_min -= 0.5
    z_max += 0.5
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title('Planar Quadrotor Animation', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Initialize plot elements
    body_line, = ax.plot([], [], 'b-', linewidth=3, label='Body')
    rotor1_circle, = ax.plot([], [], 'ro', markersize=8, label='Rotor 1')
    rotor2_circle, = ax.plot([], [], 'go', markersize=8, label='Rotor 2')
    center_point, = ax.plot([], [], 'ko', markersize=6, label='Center')
    
    # Optional: plot trajectory path
    trajectory_line, = ax.plot([], [], 'k--', linewidth=1, alpha=0.3, label='Trajectory')
    
    # Reference point and path (if ref_states is provided)
    if ref_states is not None:
        ref_point, = ax.plot([], [], 'r*', markersize=10, label='Reference')
        ref_path, = ax.plot([], [], 'r--', linewidth=1.5, alpha=0.6, label='Ref path')
    else:
        ref_point = None
        ref_path = None
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper right')
    
    # Animation function
    def animate(frame):
        """Update animation for each frame."""
        # Get current state
        x_curr = x[frame]
        z_curr = z[frame]
        theta_curr = theta[frame]
        
        # Compute current rotor positions
        cos_t = np.cos(theta_curr)
        sin_t = np.sin(theta_curr)
        
        r1_x = x_curr - l * cos_t
        r1_z = z_curr - l * sin_t
        r2_x = x_curr + l * cos_t
        r2_z = z_curr + l * sin_t
        
        # Update body line (between rotors)
        body_line.set_data([r1_x, r2_x], [r1_z, r2_z])
        
        # Update rotor positions
        rotor1_circle.set_data([r1_x], [r1_z])
        rotor2_circle.set_data([r2_x], [r2_z])
        
        # Update center point
        center_point.set_data([x_curr], [z_curr])
        
        # Update trajectory (show path up to current frame)
        trajectory_line.set_data(x[:frame+1], z[:frame+1])
        
        # Update reference point and path if provided
        if ref_states is not None:
            x_ref_curr = x_ref[frame]
            z_ref_curr = z_ref[frame]
            ref_point.set_data([x_ref_curr], [z_ref_curr])
            ref_path.set_data(x_ref[:frame+1], z_ref[:frame+1])
        
        # Update time text
        time_text.set_text(f'Time: {t[frame]:.2f} s')
        
        # Return all artists
        if ref_states is not None:
            return body_line, rotor1_circle, rotor2_circle, center_point, trajectory_line, ref_point, ref_path, time_text
        else:
            return body_line, rotor1_circle, rotor2_circle, center_point, trajectory_line, time_text
    
    # Create animation
    num_frames = len(t)
    interval = 1000 / fps  # milliseconds per frame
    
    anim = FuncAnimation(fig, animate, frames=num_frames, 
                        interval=interval, blit=False, repeat=True)
    
    # Save animation if path is provided
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        try:
            # Determine format from file extension
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            else:
                # Default to mp4
                if not save_path.endswith('.mp4'):
                    save_path += '.mp4'
                anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved successfully to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
            print("Animation will be displayed interactively instead.")
    
    return anim

