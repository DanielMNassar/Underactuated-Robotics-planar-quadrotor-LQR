# Planar Quadrotor (2D VTOL) - Underactuated Robotics Project
PowerPoint Presentation: https://lauedu74602-my.sharepoint.com/:p:/g/personal/daniel_nassar_lau_edu/ESDvqbNfJ1ZOiJ0QTE3O_IkBvhJ_HiVYqrBR5LStniAFiQ?e=FWnaf1
## 1. Project Overview

This project implements a comprehensive control and analysis framework for a **planar quadrotor** (2D vertical take-off and landing system). The planar quadrotor is a simplified 2D model of a quadrotor that moves in the vertical plane (x-z plane), making it an ideal system for studying underactuated robotics.

### Key Features

- **Nonlinear model from Newton–Euler**: Full nonlinear dynamics implementation based on first principles
- **Linearization around hover**: Analytical Jacobian computation for state-space representation
- **LQR stabilization**: Linear Quadratic Regulator for stabilizing the system around hover equilibrium
- **LQR tracking of multiple trajectories**: Time-varying reference tracking with various trajectory types
- **Wind disturbance rejection**: Analysis of controller robustness to external disturbances
- **Lyapunov stability analysis**: Mathematical proof of stability using quadratic Lyapunov functions
- **Region of Attraction (ROA) visualization**: Graphical representation of stability regions
- **Python implementation + animations**: Complete simulation framework with visualizations

### System Description

The planar quadrotor has **6 states** and **2 control inputs**, making it an underactuated system (fewer actuators than degrees of freedom). The system dynamics are derived from Newton–Euler equations in the x-z plane.

**States:**
- `x` = horizontal position (m)
- `xdot` = horizontal velocity (m/s)
- `z` = vertical position (m)
- `zdot` = vertical velocity (m/s)
- `theta` = pitch angle (radians)
- `thetadot` = angular velocity (rad/s)

**Control Inputs:**
- `FT` = total thrust = F₁ + F₂ (N)
- `tau` = torque = l · (F₂ - F₁) (N⋅m)

## 2. Project Structure

```
project_root/
    code/
        __init__.py                  # Package initialization
        dynamics.py                   # Nonlinear dynamics implementation
        simulation.py                 # Simulation engine and linearization
        controllers.py                # LQR controller design
        visualization.py              # Plotting and animation utilities
        main_lqr.py                   # LQR hover stabilization
        main_tracking_scenarios.py    # LQR tracking with multiple scenarios
        main_lqr_wind.py              # Wind disturbance simulation
        analysis_lyapunov.py          # Lyapunov stability analysis
    visuals/                          # Generated plots and MP4 animations
    README.md                         # This file
```

### File Descriptions

- **`dynamics.py`**: Implements the nonlinear dynamics equations using Newton–Euler formulation. Contains `compute_derivatives()` for state derivatives and `get_hover_condition()` for equilibrium analysis.

- **`simulation.py`**: Provides simulation engines (Euler, RK4, ODE integration) and `linearize_quadrotor()` function that computes the Jacobian matrices A and B analytically around the hover equilibrium.

- **`controllers.py`**: Contains `design_lqr_controller()` which computes the LQR gain matrix K by solving the continuous-time algebraic Riccati equation (CARE).

- **`visualization.py`**: Plotting functions for states/inputs and `animate_quadrotor()` function that creates MP4 animations of the quadrotor motion in the x-z plane.

- **`main_lqr.py`**: LQR-controlled simulation demonstrating stabilization from perturbed initial conditions back to hover equilibrium.

- **`main_tracking_scenarios.py`**: LQR tracking simulation with 5 different trajectory scenarios (smooth, sine, ramp, logistic, square wave). Change the `SCENARIO` constant (0–4) to select different trajectory types.

- **`main_lqr_wind.py`**: Demonstrates LQR controller response to unknown horizontal wind disturbance. The disturbance is added to the dynamics, not the controller.

- **`analysis_lyapunov.py`**: Computes and plots the Lyapunov function V(t) = x̃ᵀPx̃ along closed-loop trajectories, proving stability. Also generates ROA visualizations in the (x, theta) plane.

## 3. How to Run Each Module

All simulation scripts should be run from the project root directory or from within the `code/` directory.

### LQR Stabilization (`main_lqr.py`)

Demonstrates LQR control stabilizing the system from a perturbed initial condition:

```bash
cd code
python main_lqr.py
```

**What it does:**
- Designs LQR controller with default Q and R matrices
- Simulates stabilization from initial disturbance
- Generates plots showing state convergence to hover
- Creates animation saved to `../visuals/lqr_quadrotor.mp4`

**Output files:**
- `visuals/lqr_states.png` - All state trajectories
- `visuals/lqr_inputs.png` - Control input trajectories
- `visuals/lqr_stabilization.png` - Key states (x, z, theta) vs reference
- `visuals/lqr_quadrotor.mp4` - Animation of quadrotor motion

### Trajectory Tracking (`main_tracking_scenarios.py`)

Demonstrates LQR control tracking time-varying reference trajectories:

```bash
cd code
python main_tracking_scenarios.py
```

**How to choose scenarios:**
Edit the `SCENARIO` constant at the top of the file (line 29):
- `SCENARIO = 0` → Smooth S-curve transition
- `SCENARIO = 1` → Sinusoidal trajectory
- `SCENARIO = 2` → Ramp (saturated)
- `SCENARIO = 3` → Logistic S-curve
- `SCENARIO = 4` → Square wave

**Output files:**
- `visuals/tracking_scenario{N}_{name}_states.png` - States vs reference
- `visuals/tracking_scenario{N}_{name}_errors.png` - Tracking errors
- `visuals/tracking_scenario{N}_{name}_inputs.png` - Control inputs
- `visuals/tracking_scenario{N}_{name}.mp4` - Animation

### Wind Disturbance (`main_lqr_wind.py`)

Demonstrates LQR controller response to unknown horizontal wind disturbance:

```bash
cd code
python main_lqr_wind.py
```

**What it does:**
- Uses the same LQR controller as `main_lqr.py`
- Adds horizontal wind force F_wind(t) to the dynamics
- Wind profile: 0 N for t < 2s, 0.5 N for t ≥ 2s
- Controller is unaware of the disturbance (no feedforward compensation)
- Shows how LQR reacts to unmodeled disturbances

**Output files:**
- `visuals/lqr_wind_states.png` - State trajectories with disturbance
- `visuals/lqr_wind_inputs.png` - Control inputs responding to disturbance
- `visuals/lqr_quadrotor_wind.mp4` - Animation

### Lyapunov Analysis (`analysis_lyapunov.py`)

Computes and visualizes Lyapunov function V(t) and Region of Attraction:

```bash
cd code
python analysis_lyapunov.py
```

**What it does:**
- Computes V(t) = x̃ᵀPx̃ along closed-loop trajectory
- Plots V(t) to demonstrate it decreases (proving stability)
- Generates ROA visualization showing Lyapunov level sets (ellipses) in (x, theta) plane
- Shows closed-loop trajectory evolving within level sets

**Output files:**
- `visuals/lyapunov_V_t.png` - Lyapunov function V(t) vs time
- `visuals/roa_x_theta.png` - Region of Attraction in (x, theta) plane

### Generated Files Location

All plots and animations are saved to the `visuals/` directory:
- **Plots**: PNG files with 150 DPI resolution
- **Animations**: MP4 video files at 30 FPS

## 4. Description of Each Feature

### LQR Stabilization

The LQR (Linear Quadratic Regulator) controller is designed to stabilize the system around the hover equilibrium point. The controller minimizes a quadratic cost function:

**Q and R Philosophy:**
- **Q matrix**: Penalizes deviations from desired states. Default values emphasize:
  - High penalty on `z` (vertical position) and `theta` (pitch angle) to maintain altitude and attitude
  - Lower penalty on `x` (horizontal position) since horizontal drift is acceptable
- **R matrix**: Penalizes control effort. Default values are equal for thrust and torque to balance control usage.

**Closed-loop behavior:**
- The control law is: `u = u* - K(x - x*)`
- Where `u* = [m·g, 0]` is the hover control input
- The gain matrix K is computed by solving the continuous-time algebraic Riccati equation
- All states converge exponentially to the equilibrium point

**What states converge to:**
- `x → 0` (horizontal position)
- `xdot → 0` (horizontal velocity)
- `z → z*` (desired altitude, default 0.5 m)
- `zdot → 0` (vertical velocity)
- `theta → 0` (pitch angle)
- `thetadot → 0` (angular velocity)

### Trajectory Tracking

The LQR controller can track time-varying reference trajectories by extending the control law:

**Approach:**
- **Reference generator**: `x_ref(t)` provides the desired state at each time instant
- **Control law**: `u = u* - K(x - x_ref(t))`
- The same LQR gain matrix K works for all scenarios because the linearization is valid around the hover point

**Scenarios:**
1. **Smooth (0)**: Smooth S-curve transition using cosine interpolation
2. **Sine (1)**: Sinusoidal horizontal motion with amplitude 0.5 m
3. **Ramp (2)**: Constant velocity ramp (saturated at 1.0 m)
4. **Logistic (3)**: S-curve using logistic function
5. **Square wave (4)**: Step changes in horizontal position

**Why same LQR works for all:**
The LQR controller is designed around the linearized system at hover. As long as the reference trajectory keeps the system near the hover point (small angles, moderate velocities), the linearization remains valid and the same controller performs well.

### Wind Disturbance

The wind disturbance simulation demonstrates controller robustness to unmodeled external forces.

**Disturbance implementation:**
- The wind force `F_wind(t)` is added directly to the dynamics equation:
  ```
  m · ẍ = -FT · sin(θ) + F_wind(t)
  ```
- The disturbance is **not** known to the controller (no feedforward compensation)
- Wind profile: 0 N for t < 2s, then 0.5 N constant horizontal force

**How LQR reacts:**
- The controller observes the state deviation caused by the wind
- It applies corrective control inputs through the feedback law
- The system reaches a new equilibrium with a steady-state error (since there's no integral action)
- The controller maintains stability despite the unknown disturbance

### Lyapunov Analysis

The Lyapunov analysis provides a mathematical proof of stability for the LQR controller.

**Using Riccati matrix P:**
- The LQR solution provides a positive definite matrix P from the Riccati equation
- This P matrix defines a quadratic Lyapunov function: `V(x) = x̃ᵀPx̃`
- Where `x̃ = x - x*` is the error from equilibrium

**V(t) decreasing:**
- Along any closed-loop trajectory, `V(t)` is proven to decrease: `V̇(t) < 0`
- The plot of `V(t)` vs time shows monotonic decrease to zero
- This proves exponential stability of the equilibrium point

**Graphical ROA:**
- The Region of Attraction (ROA) is approximated using level sets of the Lyapunov function
- Level sets `{x : x̃ᵀPx̃ = c}` form ellipses in reduced state spaces
- The visualization shows multiple level sets in the (x, theta) plane
- The closed-loop trajectory is shown to evolve within these level sets
- Larger level sets indicate larger regions where stability is guaranteed

### Animations

The animation function provides visual feedback of the quadrotor motion in the x-z plane.

**How the animation function works:**
- Uses `matplotlib.animation.FuncAnimation` to create frame-by-frame animations
- Each frame shows:
  - **Body representation**: Blue line connecting the two rotors
  - **Rotor positions**: Red and green circles at rotor locations
  - **Center of mass**: Black dot marking the quadrotor center
  - **Trajectory path**: Dashed line showing the path traveled
  - **Time display**: Current simulation time
  - **Reference trajectory** (if provided): Red dashed line showing desired path
- The animation is saved as an MP4 video file using FFmpeg writer

**Where MP4 files are stored:**
- All animations are saved to the `visuals/` directory
- Filenames follow the pattern: `{simulation_name}_quadrotor.mp4`
- Example: `lqr_quadrotor.mp4`, `tracking_scenario1_sine.mp4`, `lqr_quadrotor_wind.mp4`

## 5. Mathematical Summary

### Nonlinear Model

The planar quadrotor dynamics are derived from Newton–Euler equations:

```
m · ẍ      = -FT · sin(θ)
m · z̈      =  FT · cos(θ) - m·g
I · θ̈      = τ
```

Where:
- `m` = mass (kg)
- `I` = moment of inertia (kg·m²)
- `g` = gravitational acceleration (m/s²)
- `FT` = total thrust (N)
- `τ` = torque (N·m)
- `θ` = pitch angle (rad)

### Linearized A, B Matrices

The system is linearized around the hover equilibrium:
- Equilibrium: `θ* = 0`, `FT* = m·g`, `τ* = 0`, all velocities zero

The linearized state-space representation is:
```
ẋ = A·x + B·u
```

Where the Jacobian matrices are computed analytically:

**A matrix** (6×6):
- Couples horizontal acceleration to pitch angle: `A[1,4] = -g`
- Standard integrator structure for position/velocity pairs

**B matrix** (6×2):
- Vertical acceleration responds to thrust: `B[3,0] = 1/m`
- Angular acceleration responds to torque: `B[5,1] = 1/I`

### LQR Cost

The LQR controller minimizes the infinite-horizon quadratic cost:

```
J = ∫₀^∞ [x̃ᵀ(t)Qx̃(t) + ũᵀ(t)Rũ(t)] dt
```

Where:
- `x̃ = x - x*` is the state error
- `ũ = u - u*` is the control error
- `Q` is the state weighting matrix (positive semi-definite)
- `R` is the input weighting matrix (positive definite)

### Control Law

The LQR control law is:

```
u = u* - K·(x - x*)
```

For tracking:

```
u = u* - K·(x - x_ref(t))
```

Where:
- `K = R⁻¹BᵀP` is the LQR gain matrix
- `P` is the solution to the continuous-time algebraic Riccati equation (CARE):
  ```
  AᵀP + PA - PBR⁻¹BᵀP + Q = 0
  ```

### Lyapunov Function

The Lyapunov function for the closed-loop system is:

```
V(x) = x̃ᵀPx̃
```

Where `P` is the Riccati solution matrix. This function:
- Is positive definite: `V(x) > 0` for all `x ≠ x*`
- Has negative definite derivative: `V̇(x) < 0` along closed-loop trajectories
- Proves exponential stability of the equilibrium point

## 6. Installation & Requirements

### Python Version

**Python 3.7 or higher** is required.

### Dependencies

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib control
```

**Package descriptions:**
- **`numpy`**: Numerical computations and array operations
- **`scipy`**: Scientific computing (ODE integration via `solve_ivp`, linear algebra)
- **`matplotlib`**: Plotting and visualization (including animation support)
- **`control`**: Control systems library (python-control) for LQR solver and Riccati equation solution

### Optional Dependencies

For MP4 animation support, FFmpeg must be installed on your system:
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
- **Linux/Mac**: Install via package manager (`apt-get install ffmpeg` or `brew install ffmpeg`)

If FFmpeg is not available, animations will still work but may need to be saved in a different format (GIF or individual frames).

### Verification

To verify the installation, run:

```bash
cd code
python -c "import numpy, scipy, matplotlib, control; print('All dependencies installed successfully!')"
```

---

## Notes

- **Original Code**: All code in this project is fully original and was not copied from any external repository.
- **Educational Purpose**: This project was built for educational purposes as part of an Underactuated Robotics course.
- **State Vector Ordering**: The state vector ordering `[x, xdot, z, zdot, theta, thetadot]` is consistent throughout the entire codebase.
- **Modular Design**: All functions are modular, well-documented with docstrings, and parameters are passed as arguments rather than hardcoded.
- **Code Quality**: The codebase is designed to be clean, maintainable, and educational.

## License

This project is for educational purposes as part of a university course.
