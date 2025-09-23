import numpy as np
from matplotlib.patches import FancyArrow
from numpy import sin, cos


class DoublePendulumParameters:
    """ Stores pendulum parameters.
    """

    def __init__(self,
                 g=9.81,
                 L_1=1.0, L_2=1.0,
                 k_1=1.0, k_2=1.0,
                 c_1=1.0, c_2=1.0,
                 m_1=1.0, m_2=1.0,
                 J_1=None, J_2=None,
                 c_w_1=1.0, c_w_2=1.0):

        """ Initialize the pendulum parameters.
        """

        # Set parameters
        self.g = g  # gravity [m/s**2]
        self.L_1 = L_1  # length of 1st segment [m]
        self.L_2 = L_2  # length of 2nd segment [m]
        self.k_1 = k_1  # torsional spring constant of 1st segment [N-m/rad]
        self.k_2 = k_2  # torsional spring constant of 2nd segment [N-m/rad]
        self.c_1 = c_1  # torsional damping constant of 1st segment [N-m-s/rad]
        self.c_2 = c_2  # torsional damping constant of 2nd segment [N-m-s/rad]
        self.m_1 = m_1  # mass of 1st segment [kg]
        self.m_2 = m_2  # mass of 2nd segment [kg]
        self.c_w_1 = c_w_1  # wind damping constant for 1st segment
        self.c_w_2 = c_w_2  # wind damping constant for 2nd segment

        # Mass-moment of inertia about the y-axis for 1st segment
        if J_1 is None:  # default to cylinder
            self.J_1 = (1.0/12.0) * self.m_1 * self.L_1
        else:
            self.J_1 = J_1

        # Mass-moment of inertia about the y-axis for 2nd segment
        if J_2 is None:  # default to cylinder
            self.J_2 = (1.0/12.0) * self.m_2 * self.L_2**2
        else:
            self.J_2 = J_2


class DoublePendulumModel:
    def __init__(self, parameters: DoublePendulumParameters = None):
        """ Initialize the pendulum model.
        """

        # Set the parameters (use defaults if none provided)
        if parameters is None:
            self.parameters = DoublePendulumParameters()
        else:
            self.parameters = parameters

        # State names
        self.state_names = ['theta_1',  # angle of 1st segment in global frame [rad]
                            'theta_2',  # angle of 2nd segment in global frame [rad]
                            'theta_dot_1',  # angular velocity of 1st segment in global frame [rad/s]
                            'theta_dot_2',  # angular velocity of 2nd segment in global frame [rad/s]
                            'x',  # x position of base [m]
                            'y',  # y position of base [m]
                            'x_dot',  # x velocity of base [m/s]
                            'y_dot',  # y velocity of base [m/s]
                            'w',  # ambient wind speed [m/s]
                            'zeta',  # ambient wind direction [rad]
                            ]

        # Input names
        self.input_names = ['x_ddot',  # x acceleration of base [m/s**2]
                            'y_ddot',  # y acceleration of base [m/s**2]
                            'tau_1',  # torque acting on 1st segment [N-m]
                            'tau_2',  # torque acting on 2nd segment [N-m]
                            ]

        # Measurement names
        self.measurement_names = ['theta_2_1']
        self.measurement_names = self.state_names + self.input_names + self.measurement_names

    def f(self, X, U):
        """ Dynamic model.
        """

        # Parameters
        p = self.parameters
        L_1, L_2 = p.L_1, p.L_2
        m_1, m_2 = p.m_1, p.m_2
        k_1, k_2 = p.k_1, p.k_2
        c_1, c_2 = p.c_1, p.c_2
        J_1, J_2 = p.J_1, p.J_2
        g = p.g
        c_w_1, c_w_2 = p.c_w_1, p.c_w_2

        # States
        theta_1, theta_2, theta_dot_1, theta_dot_2, x, y, x_dot, y_dot, w, zeta = X

        # Inputs
        x_ddot, y_ddot, tau_1, tau_2 = U

        # Angular acceleration dynamics (copied from MATLAB)
        theta_ddot_1 = (16*J_2*tau_1 - 8*J_2*k_1 - 16*J_2*c_1*theta_dot_1 - 16*J_2*c_2*theta_dot_1 + 16*J_2*c_2*theta_dot_2 - 16*J_2*k_2*theta_1 + 16*J_2*k_2*theta_2 - 2*L_2**2*k_1*m_2 + 4*L_2**2*m_2*tau_1 - 4*J_2*L_1**2*c_w_1*theta_dot_1 - 4*L_2**2*c_1*m_2*theta_dot_1 - 4*L_2**2*c_2*m_2*theta_dot_1 + 4*L_2**2*c_2*m_2*theta_dot_2 - 4*L_2**2*k_2*m_2*theta_1 + 4*L_2**2*k_2*m_2*theta_2 - 2*L_1*L_2**3*m_2**2*theta_dot_2**2*sin(theta_1 - theta_2) + 2*L_1*L_2**2*g*m_2**2*sin(theta_1) - 2*L_1*L_2**2*m_2**2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*m_2**2*y_ddot*sin(theta_1) + 2*L_1*L_2**2*g*m_2**2*sin(theta_1 - 2*theta_2) + 2*L_1*L_2**2*m_2**2*x_ddot*cos(theta_1 - 2*theta_2) - 8*J_2*L_1*c_w_1*x_dot*cos(theta_1) + 8*J_2*L_1*g*m_1*sin(theta_1) + 16*J_2*L_1*g*m_2*sin(theta_1) - 8*J_2*L_1*m_1*x_ddot*cos(theta_1) - 16*J_2*L_1*m_2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*m_2**2*y_ddot*sin(theta_1 - 2*theta_2) + 8*J_2*L_1*c_w_1*y_dot*sin(theta_1) + 8*J_2*L_1*m_1*y_ddot*sin(theta_1) + 16*J_2*L_1*m_2*y_ddot*sin(theta_1) - 2*L_1**2*L_2**2*m_2**2*theta_dot_1**2*sin(2*theta_1 - 2*theta_2) - 8*L_1*L_2*m_2*tau_2*cos(theta_1 - theta_2) - 8*J_2*L_1*c_w_1*w*sin(theta_1 - zeta) - L_1**2*L_2**2*c_w_1*m_2*theta_dot_1 + 2*L_1**2*L_2**2*c_w_2*m_2*theta_dot_1 - 2*L_1*L_2**2*c_w_2*m_2*w*sin(theta_1 - 2*theta_2 + zeta) - 8*L_1*L_2*c_2*m_2*theta_dot_1*cos(theta_1 - theta_2) + 8*L_1*L_2*c_2*m_2*theta_dot_2*cos(theta_1 - theta_2) - 8*L_1*L_2*k_2*m_2*theta_1*cos(theta_1 - theta_2) + 8*L_1*L_2*k_2*m_2*theta_2*cos(theta_1 - theta_2) + 2*L_1**2*L_2**2*c_w_2*m_2*theta_dot_1*cos(2*theta_1 - 2*theta_2) - 2*L_1*L_2**2*c_w_1*m_2*x_dot*cos(theta_1) + 2*L_1*L_2**2*c_w_2*m_2*x_dot*cos(theta_1) + 2*L_1*L_2**2*g*m_1*m_2*sin(theta_1) - 2*L_1*L_2**2*m_1*m_2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*c_w_1*m_2*y_dot*sin(theta_1) - 2*L_1*L_2**2*c_w_2*m_2*y_dot*sin(theta_1) + 2*L_1*L_2**2*m_1*m_2*y_ddot*sin(theta_1) - 8*J_2*L_1*L_2*m_2*theta_dot_2**2*sin(theta_1 - theta_2) + 2*L_1*L_2**3*c_w_2*m_2*theta_dot_2*cos(theta_1 - theta_2) + 2*L_1*L_2**2*c_w_2*m_2*x_dot*cos(theta_1 - 2*theta_2) + 2*L_1*L_2**2*c_w_2*m_2*y_dot*sin(theta_1 - 2*theta_2) - 2*L_1*L_2**2*c_w_1*m_2*w*sin(theta_1 - zeta) + 2*L_1*L_2**2*c_w_2*m_2*w*sin(theta_1 - zeta))/(16*J_1*J_2 + 2*L_1**2*L_2**2*m_2**2 + 4*J_2*L_1**2*m_1 + 4*J_1*L_2**2*m_2 + 16*J_2*L_1**2*m_2 + L_1**2*L_2**2*m_1*m_2 - 2*L_1**2*L_2**2*m_2**2*cos(2*theta_1 - 2*theta_2))
        theta_ddot_2 = (16*J_1*tau_2 + 16*J_1*c_2*theta_dot_1 - 16*J_1*c_2*theta_dot_2 + 16*J_1*k_2*theta_1 - 16*J_1*k_2*theta_2 + 4*L_1**2*m_1*tau_2 + 16*L_1**2*m_2*tau_2 - 4*J_1*L_2**2*c_w_2*theta_dot_2 + 4*L_1**2*c_2*m_1*theta_dot_1 - 4*L_1**2*c_2*m_1*theta_dot_2 + 16*L_1**2*c_2*m_2*theta_dot_1 - 16*L_1**2*c_2*m_2*theta_dot_2 + 4*L_1**2*k_2*m_1*theta_1 - 4*L_1**2*k_2*m_1*theta_2 + 16*L_1**2*k_2*m_2*theta_1 - 16*L_1**2*k_2*m_2*theta_2 - 4*L_1**2*L_2*g*m_2**2*sin(2*theta_1 - theta_2) + 4*L_1**2*L_2*m_2**2*x_ddot*cos(2*theta_1 - theta_2) + 8*L_1**3*L_2*m_2**2*theta_dot_1**2*sin(theta_1 - theta_2) - 4*L_1**2*L_2*m_2**2*y_ddot*sin(2*theta_1 - theta_2) + 4*L_1**2*L_2*g*m_2**2*sin(theta_2) - 4*L_1**2*L_2*m_2**2*x_ddot*cos(theta_2) + 4*L_1**2*L_2*m_2**2*y_ddot*sin(theta_2) - 8*J_1*L_2*c_w_2*x_dot*cos(theta_2) + 8*J_1*L_2*g*m_2*sin(theta_2) - 8*J_1*L_2*m_2*x_ddot*cos(theta_2) + 8*J_1*L_2*c_w_2*y_dot*sin(theta_2) + 8*J_1*L_2*m_2*y_ddot*sin(theta_2) + 2*L_1**2*L_2**2*m_2**2*theta_dot_2**2*sin(2*theta_1 - 2*theta_2) + 4*L_1*L_2*k_1*m_2*cos(theta_1 - theta_2) - 8*L_1*L_2*m_2*tau_1*cos(theta_1 - theta_2) - 8*J_1*L_2*c_w_2*w*sin(theta_2 - zeta) - L_1**2*L_2**2*c_w_2*m_1*theta_dot_2 - 4*L_1**2*L_2**2*c_w_2*m_2*theta_dot_2 + 2*L_1**2*L_2*c_w_1*m_2*x_dot*cos(2*theta_1 - theta_2) - 2*L_1**2*L_2*g*m_1*m_2*sin(2*theta_1 - theta_2) + 2*L_1**2*L_2*m_1*m_2*x_ddot*cos(2*theta_1 - theta_2) - 2*L_1**2*L_2*c_w_1*m_2*y_dot*sin(2*theta_1 - theta_2) + 2*L_1**3*L_2*m_1*m_2*theta_dot_1**2*sin(theta_1 - theta_2) - 8*J_1*L_1*L_2*c_w_2*theta_dot_1*cos(theta_1 - theta_2) - 2*L_1**2*L_2*m_1*m_2*y_ddot*sin(2*theta_1 - theta_2) - 2*L_1**2*L_2*c_w_1*m_2*w*sin(theta_2 - 2*theta_1 + zeta) + 8*L_1*L_2*c_1*m_2*theta_dot_1*cos(theta_1 - theta_2) + 8*L_1*L_2*c_2*m_2*theta_dot_1*cos(theta_1 - theta_2) - 8*L_1*L_2*c_2*m_2*theta_dot_2*cos(theta_1 - theta_2) + 8*L_1*L_2*k_2*m_2*theta_1*cos(theta_1 - theta_2) - 8*L_1*L_2*k_2*m_2*theta_2*cos(theta_1 - theta_2) + 2*L_1**2*L_2*c_w_1*m_2*x_dot*cos(theta_2) - 2*L_1**2*L_2*c_w_2*m_1*x_dot*cos(theta_2) - 8*L_1**2*L_2*c_w_2*m_2*x_dot*cos(theta_2) - 2*L_1**2*L_2*c_w_1*m_2*y_dot*sin(theta_2) + 2*L_1**2*L_2*c_w_2*m_1*y_dot*sin(theta_2) + 8*L_1**2*L_2*c_w_2*m_2*y_dot*sin(theta_2) + 8*J_1*L_1*L_2*m_2*theta_dot_1**2*sin(theta_1 - theta_2) + 2*L_1**3*L_2*c_w_1*m_2*theta_dot_1*cos(theta_1 - theta_2) - 2*L_1**3*L_2*c_w_2*m_1*theta_dot_1*cos(theta_1 - theta_2) - 8*L_1**3*L_2*c_w_2*m_2*theta_dot_1*cos(theta_1 - theta_2) + 2*L_1**2*L_2*c_w_1*m_2*w*sin(theta_2 - zeta) - 2*L_1**2*L_2*c_w_2*m_1*w*sin(theta_2 - zeta) - 8*L_1**2*L_2*c_w_2*m_2*w*sin(theta_2 - zeta))/(16*J_1*J_2 + 2*L_1**2*L_2**2*m_2**2 + 4*J_2*L_1**2*m_1 + 4*J_1*L_2**2*m_2 + 16*J_2*L_1**2*m_2 + L_1**2*L_2**2*m_1*m_2 - 2*L_1**2*L_2**2*m_2**2*cos(2*theta_1 - 2*theta_2))

        # Wind dynamics
        w_dot = 0.0
        zeta_dot = 0.0

        # Package and return xdot
        x_dot = [theta_dot_1, theta_dot_2,
                 theta_ddot_1, theta_ddot_2,
                 x_dot, y_dot,
                 x_ddot, y_ddot,
                 w_dot, zeta_dot]

        return x_dot

    def h(self, X, U):
        """ Measurement model.
        """

        # States
        theta_1, theta_2, theta_dot_1, theta_dot_2, x, y, x_dot, y_dot, w, zeta = X

        # Inputs
        x_ddot, y_ddot, tau_1, tau_2 = U

        # Dynamics
        theta_dot_1, theta_dot_2, theta_ddot_1, theta_ddot_2, x_dot, y_dot, x_ddot, y_ddot, w_dot, zeta_dot = (
            self.f(X, U))

        # Relative angle of 2nd segment
        theta_2_1 = theta_2 - theta_1

        Y = (list(X) + list(U) + [theta_2_1])
        return Y


class DoublePendulumDrawer:
    """
    Draw a double pendulum with a moving base on a given axis.
    Each segment has its own color and optional trail.
    """

    def __init__(self, base_x, base_y, theta1, theta2,
                 L1=1.0, L2=1.0,
                 seg1_kwargs=None, seg2_kwargs=None,
                 trail1=False, trail2=False, trail_base=False):
        """
        Parameters
        ----------
        base_x, base_y : array-like
            Arrays of base positions over time.
        theta1, theta2 : array-like
            Arrays of angles of first and second segments (rad).
        L1, L2 : float
            Lengths of the pendulum arms.
        seg1_kwargs, seg2_kwargs : dict, optional
            Keyword arguments for plotting segment 1 and segment 2
            (passed to `ax.plot`).
        trail1, trail2 : bool
            If True, show trailing paths for the respective segment.
        trail_base : bool
            If True, show trail of the moving base in black.
        """
        self.base_x = np.asarray(base_x)
        self.base_y = np.asarray(base_y)
        self.theta1 = np.asarray(theta1)
        self.theta2 = np.asarray(theta2)
        self.L1 = L1
        self.L2 = L2
        self.trail1 = trail1
        self.trail2 = trail2
        self.trail_base = trail_base

        # Default plotting kwargs
        default_seg1 = dict(linewidth=3, color="blue", markersize=8, markerfacecolor="black")
        default_seg2 = dict(linewidth=3, color="red", markersize=8, markerfacecolor="black")

        # Allow user overrides
        self.seg1_kwargs = {**default_seg1, **(seg1_kwargs or {})}
        self.seg2_kwargs = {**default_seg2, **(seg2_kwargs or {})}

        # Compute joint positions
        self.x1 = self.base_x + L1 * np.sin(self.theta1)
        self.y1 = self.base_y + L1 * np.cos(self.theta1)
        self.x2 = self.x1 + L2 * np.sin(self.theta2)
        self.y2 = self.y1 + L2 * np.cos(self.theta2)

    def draw(self, ax, frame=-1):
        """
        Draw the pendulum at a given frame on the provided axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to draw on.
        frame : int, optional
            Index of the frame to draw. Default -1 (last frame).
        """
        if frame < 0:
            frame = len(self.base_x) + frame  # allow negative indexing

        # Segment 1
        x_seg1 = [self.base_x[frame], self.x1[frame]]
        y_seg1 = [self.base_y[frame], self.y1[frame]]
        ax.plot(x_seg1, y_seg1, 'o-', **self.seg1_kwargs)

        # Segment 2
        x_seg2 = [self.x1[frame], self.x2[frame]]
        y_seg2 = [self.y1[frame], self.y2[frame]]
        ax.plot(x_seg2, y_seg2, 'o-', **self.seg2_kwargs)

        # Trails
        if self.trail1:
            ax.plot(self.x1[:frame+1], self.y1[:frame+1],
                    '-', lw=1, color=self.seg1_kwargs.get("color", "blue"), alpha=0.5)
        if self.trail2:
            ax.plot(self.x2[:frame+1], self.y2[:frame+1],
                    '-', lw=1, color=self.seg2_kwargs.get("color", "red"), alpha=0.5)
        if self.trail_base:
            ax.plot(self.base_x[:frame+1], self.base_y[:frame+1],
                    '-', lw=1, color='black', alpha=0.5)

    def get_axis_bounds(self, margin=0.1):
        """
        Compute axis bounds that fit the full pendulum motion.

        Parameters
        ----------
        margin : float, optional
            Fractional padding to add around the min/max bounds.

        Returns
        -------
        (xmin, xmax, ymin, ymax) : tuple of floats
            Recommended axis limits for plotting.
        """
        # Collect all x and y coordinates
        xs = np.concatenate([self.base_x, self.x1, self.x2])
        ys = np.concatenate([self.base_y, self.y1, self.y2])

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        # Add margin
        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= margin * x_range
        xmax += margin * x_range
        ymin -= margin * y_range
        ymax += margin * y_range

        return xmin, xmax, ymin, ymax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

class ArrowDrawer:
    """
    Draws an arrow with given magnitude and direction.
    Angle: 0 = up (y-axis), positive = clockwise.
    Accepts kwargs for FancyArrow styling.
    """

    def __init__(self, origin=(0, 0), magnitude=1.0, angle=0.0, **kwargs):
        self.origin = np.array(origin, dtype=float)
        self.magnitude = magnitude
        self.angle = angle
        self.kwargs = kwargs
        self.arrow = None

    def draw(self, ax, magnitude=None, angle=None, origin=None):
        # Update parameters if provided
        if magnitude is not None:
            self.magnitude = magnitude
        if angle is not None:
            self.angle = angle
        if origin is not None:
            self.origin = np.array(origin, dtype=float)

        # Remove old arrow
        if self.arrow is not None:
            self.arrow.remove()

        # Compute dx, dy using 0 = up, positive = clockwise
        dx = self.magnitude * np.sin(self.angle)
        dy = self.magnitude * np.cos(self.angle)

        # Draw new arrow
        self.arrow = FancyArrow(
            *self.origin,
            dx,
            dy,
            **self.kwargs
        )
        ax.add_patch(self.arrow)
        return [self.arrow]

    def set_style(self, **kwargs):
        self.kwargs.update(kwargs)

    def get_end_point(self):
        dx = self.magnitude * np.sin(self.angle)
        dy = self.magnitude * np.cos(self.angle)
        return self.origin + np.array([dx, dy])

class DualAntennaPendulumDrawer:
    """
    Draw two double pendulums (like fly antennae) sharing the same moving base.
    Uses a single dot to represent the fly body instead of a connecting bar.
    """

    def __init__(self, base_x, base_y, 
                 theta1_left, theta2_left, theta1_right, theta2_right,
                 L1_left=1.0, L2_left=1.0, L1_right=1.0, L2_right=1.0,
                 left_seg1_kwargs=None, left_seg2_kwargs=None,
                 right_seg1_kwargs=None, right_seg2_kwargs=None,
                 trail1_left=False, trail2_left=False, 
                 trail1_right=False, trail2_right=False, 
                 trail_base=False,
                 antenna_separation=0.2,
                 body_style='dot',  # NEW: 'dot', 'bar', or 'none'
                 body_kwargs=None):  # NEW: styling for the body
        
        self.base_x = np.asarray(base_x)
        self.base_y = np.asarray(base_y)
        
        # Left antenna
        self.theta1_left = np.asarray(theta1_left)
        self.theta2_left = np.asarray(theta2_left)
        self.L1_left = L1_left
        self.L2_left = L2_left
        
        # Right antenna
        self.theta1_right = np.asarray(theta1_right)
        self.theta2_right = np.asarray(theta2_right)
        self.L1_right = L1_right
        self.L2_right = L2_right
        
        self.antenna_separation = antenna_separation
        self.trail1_left = trail1_left
        self.trail2_left = trail2_left
        self.trail1_right = trail1_right
        self.trail2_right = trail2_right
        self.trail_base = trail_base
        
        # NEW: Body styling options
        self.body_style = body_style
        default_body = dict(markersize=12, color='black', markerfacecolor='darkgray', 
                           markeredgecolor='black', markeredgewidth=2)
        self.body_kwargs = {**default_body, **(body_kwargs or {})}

        # Default styling - make them look like antennae
        default_left_seg1 = dict(linewidth=3, color="darkblue", markersize=6, markerfacecolor="black")
        default_left_seg2 = dict(linewidth=2, color="blue", markersize=4, markerfacecolor="darkblue")
        default_right_seg1 = dict(linewidth=3, color="darkred", markersize=6, markerfacecolor="black")
        default_right_seg2 = dict(linewidth=2, color="red", markersize=4, markerfacecolor="darkred")

        self.left_seg1_kwargs = {**default_left_seg1, **(left_seg1_kwargs or {})}
        self.left_seg2_kwargs = {**default_left_seg2, **(left_seg2_kwargs or {})}
        self.right_seg1_kwargs = {**default_right_seg1, **(right_seg1_kwargs or {})}
        self.right_seg2_kwargs = {**default_right_seg2, **(right_seg2_kwargs or {})}

        # Compute antenna base positions (offset from main base)
        self.base_x_left = self.base_x - self.antenna_separation / 2
        self.base_y_left = self.base_y
        self.base_x_right = self.base_x + self.antenna_separation / 2
        self.base_y_right = self.base_y

        # Compute joint positions for left antenna
        self.x1_left = self.base_x_left + self.L1_left * np.sin(self.theta1_left)
        self.y1_left = self.base_y_left + self.L1_left * np.cos(self.theta1_left)
        self.x2_left = self.x1_left + self.L2_left * np.sin(self.theta2_left)
        self.y2_left = self.y1_left + self.L2_left * np.cos(self.theta2_left)

        # Compute joint positions for right antenna
        self.x1_right = self.base_x_right + self.L1_right * np.sin(self.theta1_right)
        self.y1_right = self.base_y_right + self.L1_right * np.cos(self.theta1_right)
        self.x2_right = self.x1_right + self.L2_right * np.sin(self.theta2_right)
        self.y2_right = self.y1_right + self.L2_right * np.cos(self.theta2_right)

    def draw(self, ax, frame=-1):
        """Draw both antennae at the given frame."""
        if frame < 0:
            frame = len(self.base_x) + frame

        # Draw left antenna
        x_seg1_left = [self.base_x_left[frame], self.x1_left[frame]]
        y_seg1_left = [self.base_y_left[frame], self.y1_left[frame]]
        ax.plot(x_seg1_left, y_seg1_left, 'o-', **self.left_seg1_kwargs)

        x_seg2_left = [self.x1_left[frame], self.x2_left[frame]]
        y_seg2_left = [self.y1_left[frame], self.y2_left[frame]]
        ax.plot(x_seg2_left, y_seg2_left, 'o-', **self.left_seg2_kwargs)

        # Draw right antenna
        x_seg1_right = [self.base_x_right[frame], self.x1_right[frame]]
        y_seg1_right = [self.base_y_right[frame], self.y1_right[frame]]
        ax.plot(x_seg1_right, y_seg1_right, 'o-', **self.right_seg1_kwargs)

        x_seg2_right = [self.x1_right[frame], self.x2_right[frame]]
        y_seg2_right = [self.y1_right[frame], self.y2_right[frame]]
        ax.plot(x_seg2_right, y_seg2_right, 'o-', **self.right_seg2_kwargs)

        # NEW: Draw body based on style
        if self.body_style == 'dot':
            # Single dot at the center of the base
            ax.plot(self.base_x[frame], self.base_y[frame], 'o', **self.body_kwargs)
            
        elif self.body_style == 'bar':
            # Original connecting bar
            ax.plot([self.base_x_left[frame], self.base_x_right[frame]], 
                   [self.base_y_left[frame], self.base_y_right[frame]], 
                   'o-', linewidth=4, color='black', markersize=8, markerfacecolor='gray')
                   
        elif self.body_style == 'none':
            # No body representation
            pass

        # Trails
        if self.trail1_left:
            ax.plot(self.x1_left[:frame+1], self.y1_left[:frame+1],
                    '-', lw=1, color=self.left_seg1_kwargs.get("color", "darkblue"), alpha=0.5)
        if self.trail2_left:
            ax.plot(self.x2_left[:frame+1], self.y2_left[:frame+1],
                    '-', lw=1, color=self.left_seg2_kwargs.get("color", "blue"), alpha=0.5)
        if self.trail1_right:
            ax.plot(self.x1_right[:frame+1], self.y1_right[:frame+1],
                    '-', lw=1, color=self.right_seg1_kwargs.get("color", "darkred"), alpha=0.5)
        if self.trail2_right:
            ax.plot(self.x2_right[:frame+1], self.y2_right[:frame+1],
                    '-', lw=1, color=self.right_seg2_kwargs.get("color", "red"), alpha=0.5)
        if self.trail_base:
            ax.plot(self.base_x[:frame+1], self.base_y[:frame+1],
                    '-', lw=1, color='black', alpha=0.5)

    def get_axis_bounds(self, margin=0.1):
        """Compute axis bounds for both antennae."""
        xs = np.concatenate([
            self.base_x_left, self.x1_left, self.x2_left,
            self.base_x_right, self.x1_right, self.x2_right
        ])
        ys = np.concatenate([
            self.base_y_left, self.y1_left, self.y2_left,
            self.base_y_right, self.y1_right, self.y2_right
        ])

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        x_range = xmax - xmin
        y_range = ymax - ymin
        xmin -= margin * x_range
        xmax += margin * x_range
        ymin -= margin * y_range
        ymax += margin * y_range

        return xmin, xmax, ymin, ymax


class DualAntennaParameters:
    """ Stores parameters for dual antenna system - same as before """
    
    def __init__(self,
                 # Global parameters
                 g=9.81,
                 
                 # Left antenna parameters
                 L_1_l=1.0, L_2_l=1.0,
                 k_1_l=1.0, k_2_l=1.0,
                 c_1_l=1.0, c_2_l=1.0,
                 m_1_l=1.0, m_2_l=1.0,
                 J_1_l=None, J_2_l=None,
                 c_w_1_l=1.0, c_w_2_l=1.0,
                 
                 # Right antenna parameters
                 L_1_r=1.0, L_2_r=1.0,
                 k_1_r=1.0, k_2_r=1.0,
                 c_1_r=1.0, c_2_r=1.0,
                 m_1_r=1.0, m_2_r=1.0,
                 J_1_r=None, J_2_r=None,
                 c_w_1_r=1.0, c_w_2_r=1.0):

        # Global parameters
        self.g = g
        
        # Left antenna parameters
        self.L_1_l = L_1_l
        self.L_2_l = L_2_l
        self.k_1_l = k_1_l
        self.k_2_l = k_2_l
        self.c_1_l = c_1_l
        self.c_2_l = c_2_l
        self.m_1_l = m_1_l
        self.m_2_l = m_2_l
        self.c_w_1_l = c_w_1_l
        self.c_w_2_l = c_w_2_l
        
        # Right antenna parameters
        self.L_1_r = L_1_r
        self.L_2_r = L_2_r
        self.k_1_r = k_1_r
        self.k_2_r = k_2_r
        self.c_1_r = c_1_r
        self.c_2_r = c_2_r
        self.m_1_r = m_1_r
        self.m_2_r = m_2_r
        self.c_w_1_r = c_w_1_r
        self.c_w_2_r = c_w_2_r

        # Mass-moment of inertia for left antenna
        if J_1_l is None:
            self.J_1_l = (1.0/12.0) * self.m_1_l * self.L_1_l**2
        else:
            self.J_1_l = J_1_l

        if J_2_l is None:
            self.J_2_l = (1.0/12.0) * self.m_2_l * self.L_2_l**2
        else:
            self.J_2_l = J_2_l
            
        # Mass-moment of inertia for right antenna
        if J_1_r is None:
            self.J_1_r = (1.0/12.0) * self.m_1_r * self.L_1_r**2
        else:
            self.J_1_r = J_1_r

        if J_2_r is None:
            self.J_2_r = (1.0/12.0) * self.m_2_r * self.L_2_r**2
        else:
            self.J_2_r = J_2_r

import numpy as np
from numpy import sin, cos


class DualAntennaParameters:
    """ Stores parameters for dual antenna system.
    Each antenna has its own set of parameters.
    """

    def __init__(self,
                 # Global parameters
                 g=9.81,
                 
                 # Left antenna parameters
                 L_1_l=1.0, L_2_l=1.0,
                 k_1_l=1.0, k_2_l=1.0,
                 c_1_l=1.0, c_2_l=1.0,
                 m_1_l=1.0, m_2_l=1.0,
                 J_1_l=None, J_2_l=None,
                 c_w_1_l=1.0, c_w_2_l=1.0,
                 
                 # Right antenna parameters
                 L_1_r=1.0, L_2_r=1.0,
                 k_1_r=1.0, k_2_r=1.0,
                 c_1_r=1.0, c_2_r=1.0,
                 m_1_r=1.0, m_2_r=1.0,
                 J_1_r=None, J_2_r=None,
                 c_w_1_r=1.0, c_w_2_r=1.0):

        # Global parameters
        self.g = g  # gravity [m/s**2]
        
        # Left antenna parameters
        self.L_1_l = L_1_l  # length of 1st segment [m]
        self.L_2_l = L_2_l  # length of 2nd segment [m]
        self.k_1_l = k_1_l  # torsional spring constant of 1st segment [N-m/rad]
        self.k_2_l = k_2_l  # torsional spring constant of 2nd segment [N-m/rad]
        self.c_1_l = c_1_l  # torsional damping constant of 1st segment [N-m-s/rad]
        self.c_2_l = c_2_l  # torsional damping constant of 2nd segment [N-m-s/rad]
        self.m_1_l = m_1_l  # mass of 1st segment [kg]
        self.m_2_l = m_2_l  # mass of 2nd segment [kg]
        self.c_w_1_l = c_w_1_l  # wind damping constant for 1st segment
        self.c_w_2_l = c_w_2_l  # wind damping constant for 2nd segment
        
        # Right antenna parameters
        self.L_1_r = L_1_r  # length of 1st segment [m]
        self.L_2_r = L_2_r  # length of 2nd segment [m]
        self.k_1_r = k_1_r  # torsional spring constant of 1st segment [N-m/rad]
        self.k_2_r = k_2_r  # torsional spring constant of 2nd segment [N-m/rad]
        self.c_1_r = c_1_r  # torsional damping constant of 1st segment [N-m-s/rad]
        self.c_2_r = c_2_r  # torsional damping constant of 2nd segment [N-m-s/rad]
        self.m_1_r = m_1_r  # mass of 1st segment [kg]
        self.m_2_r = m_2_r  # mass of 2nd segment [kg]
        self.c_w_1_r = c_w_1_r  # wind damping constant for 1st segment
        self.c_w_2_r = c_w_2_r  # wind damping constant for 2nd segment

        # Mass-moment of inertia for left antenna
        if J_1_l is None:  # default to cylinder
            self.J_1_l = (1.0/12.0) * self.m_1_l * self.L_1_l**2
        else:
            self.J_1_l = J_1_l

        if J_2_l is None:  # default to cylinder
            self.J_2_l = (1.0/12.0) * self.m_2_l * self.L_2_l**2
        else:
            self.J_2_l = J_2_l
            
        # Mass-moment of inertia for right antenna
        if J_1_r is None:  # default to cylinder
            self.J_1_r = (1.0/12.0) * self.m_1_r * self.L_1_r**2
        else:
            self.J_1_r = J_1_r

        if J_2_r is None:  # default to cylinder
            self.J_2_r = (1.0/12.0) * self.m_2_r * self.L_2_r**2
        else:
            self.J_2_r = J_2_r


class DualAntennaModel:
    def __init__(self, parameters: DualAntennaParameters = None):
        """ Initialize the dual antenna model with individual control for each antenna.
        """

        # Set the parameters (use defaults if none provided)
        if parameters is None:
            self.parameters = DualAntennaParameters()
        else:
            self.parameters = parameters

        # State names for the combined system
        self.state_names = [
            # Left antenna states
            'theta_1_l',  # angle of 1st segment of left antenna [rad]
            'theta_2_l',  # angle of 2nd segment of left antenna [rad]
            'theta_dot_1_l',  # angular velocity of 1st segment of left antenna [rad/s]
            'theta_dot_2_l',  # angular velocity of 2nd segment of left antenna [rad/s]
            'x_l',  # x position of left antenna base [m]
            'y_l',  # y position of left antenna base [m]
            'x_dot_l',  # x velocity of left antenna base [m/s]
            'y_dot_l',  # y velocity of left antenna base [m/s]
            'w_l',  # ambient wind speed for left antenna [m/s]
            'zeta_l',  # ambient wind direction for left antenna [rad]
            
            # Right antenna states
            'theta_1_r',  # angle of 1st segment of right antenna [rad]
            'theta_2_r',  # angle of 2nd segment of right antenna [rad]
            'theta_dot_1_r',  # angular velocity of 1st segment of right antenna [rad/s]
            'theta_dot_2_r',  # angular velocity of 2nd segment of right antenna [rad/s]
            'x_r',  # x position of right antenna base [m]
            'y_r',  # y position of right antenna base [m]
            'x_dot_r',  # x velocity of right antenna base [m/s]
            'y_dot_r',  # y velocity of right antenna base [m/s]
            'w_r',  # ambient wind speed for right antenna [m/s]
            'zeta_r',  # ambient wind direction for right antenna [rad]
        ]

        # Input names for the combined system
        self.input_names = [
            # Left antenna inputs
            'x_ddot_l',  # x acceleration of left antenna base [m/s**2]
            'y_ddot_l',  # y acceleration of left antenna base [m/s**2]
            'tau_1_l',  # torque acting on 1st segment of left antenna [N-m]
            'tau_2_l',  # torque acting on 2nd segment of left antenna [N-m]
            
            # Right antenna inputs
            'x_ddot_r',  # x acceleration of right antenna base [m/s**2]
            'y_ddot_r',  # y acceleration of right antenna base [m/s**2]
            'tau_1_r',  # torque acting on 1st segment of right antenna [N-m]
            'tau_2_r',  # torque acting on 2nd segment of right antenna [N-m]
        ]

        # Measurement names
        self.measurement_names = ['theta_2_1_l', 'theta_2_1_r']
        self.measurement_names = self.state_names + self.input_names + self.measurement_names

    def f_single_antenna(self, X_single, U_single, antenna_params):
        """ Dynamic model for a single antenna (same as original double pendulum).
        
        Parameters:
        -----------
        X_single : array-like
            State vector for single antenna: [theta_1, theta_2, theta_dot_1, theta_dot_2, x, y, x_dot, y_dot, w, zeta]
        U_single : array-like  
            Input vector for single antenna: [x_ddot, y_ddot, tau_1, tau_2]
        antenna_params : dict
            Parameters for this specific antenna
        """
        
        # Extract parameters for this antenna
        L_1, L_2 = antenna_params['L_1'], antenna_params['L_2']
        m_1, m_2 = antenna_params['m_1'], antenna_params['m_2']
        k_1, k_2 = antenna_params['k_1'], antenna_params['k_2']
        c_1, c_2 = antenna_params['c_1'], antenna_params['c_2']
        J_1, J_2 = antenna_params['J_1'], antenna_params['J_2']
        g = antenna_params['g']
        c_w_1, c_w_2 = antenna_params['c_w_1'], antenna_params['c_w_2']

        # States
        theta_1, theta_2, theta_dot_1, theta_dot_2, x, y, x_dot, y_dot, w, zeta = X_single

        # Inputs
        x_ddot, y_ddot, tau_1, tau_2 = U_single

        # Angular acceleration dynamics (same as original)
        theta_ddot_1 = (16*J_2*tau_1 - 8*J_2*k_1 - 16*J_2*c_1*theta_dot_1 - 16*J_2*c_2*theta_dot_1 + 16*J_2*c_2*theta_dot_2 - 16*J_2*k_2*theta_1 + 16*J_2*k_2*theta_2 - 2*L_2**2*k_1*m_2 + 4*L_2**2*m_2*tau_1 - 4*J_2*L_1**2*c_w_1*theta_dot_1 - 4*L_2**2*c_1*m_2*theta_dot_1 - 4*L_2**2*c_2*m_2*theta_dot_1 + 4*L_2**2*c_2*m_2*theta_dot_2 - 4*L_2**2*k_2*m_2*theta_1 + 4*L_2**2*k_2*m_2*theta_2 - 2*L_1*L_2**3*m_2**2*theta_dot_2**2*sin(theta_1 - theta_2) + 2*L_1*L_2**2*g*m_2**2*sin(theta_1) - 2*L_1*L_2**2*m_2**2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*m_2**2*y_ddot*sin(theta_1) + 2*L_1*L_2**2*g*m_2**2*sin(theta_1 - 2*theta_2) + 2*L_1*L_2**2*m_2**2*x_ddot*cos(theta_1 - 2*theta_2) - 8*J_2*L_1*c_w_1*x_dot*cos(theta_1) + 8*J_2*L_1*g*m_1*sin(theta_1) + 16*J_2*L_1*g*m_2*sin(theta_1) - 8*J_2*L_1*m_1*x_ddot*cos(theta_1) - 16*J_2*L_1*m_2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*m_2**2*y_ddot*sin(theta_1 - 2*theta_2) + 8*J_2*L_1*c_w_1*y_dot*sin(theta_1) + 8*J_2*L_1*m_1*y_ddot*sin(theta_1) + 16*J_2*L_1*m_2*y_ddot*sin(theta_1) - 2*L_1**2*L_2**2*m_2**2*theta_dot_1**2*sin(2*theta_1 - 2*theta_2) - 8*L_1*L_2*m_2*tau_2*cos(theta_1 - theta_2) - 8*J_2*L_1*c_w_1*w*sin(theta_1 - zeta) - L_1**2*L_2**2*c_w_1*m_2*theta_dot_1 + 2*L_1**2*L_2**2*c_w_2*m_2*theta_dot_1 - 2*L_1*L_2**2*c_w_2*m_2*w*sin(theta_1 - 2*theta_2 + zeta) - 8*L_1*L_2*c_2*m_2*theta_dot_1*cos(theta_1 - theta_2) + 8*L_1*L_2*c_2*m_2*theta_dot_2*cos(theta_1 - theta_2) - 8*L_1*L_2*k_2*m_2*theta_1*cos(theta_1 - theta_2) + 8*L_1*L_2*k_2*m_2*theta_2*cos(theta_1 - theta_2) + 2*L_1**2*L_2**2*c_w_2*m_2*theta_dot_1*cos(2*theta_1 - 2*theta_2) - 2*L_1*L_2**2*c_w_1*m_2*x_dot*cos(theta_1) + 2*L_1*L_2**2*c_w_2*m_2*x_dot*cos(theta_1) + 2*L_1*L_2**2*g*m_1*m_2*sin(theta_1) - 2*L_1*L_2**2*m_1*m_2*x_ddot*cos(theta_1) + 2*L_1*L_2**2*c_w_1*m_2*y_dot*sin(theta_1) - 2*L_1*L_2**2*c_w_2*m_2*y_dot*sin(theta_1) + 2*L_1*L_2**2*m_1*m_2*y_ddot*sin(theta_1) - 8*J_2*L_1*L_2*m_2*theta_dot_2**2*sin(theta_1 - theta_2) + 2*L_1*L_2**3*c_w_2*m_2*theta_dot_2*cos(theta_1 - theta_2) + 2*L_1*L_2**2*c_w_2*m_2*x_dot*cos(theta_1 - 2*theta_2) + 2*L_1*L_2**2*c_w_2*m_2*y_dot*sin(theta_1 - 2*theta_2) - 2*L_1*L_2**2*c_w_1*m_2*w*sin(theta_1 - zeta) + 2*L_1*L_2**2*c_w_2*m_2*w*sin(theta_1 - zeta))/(16*J_1*J_2 + 2*L_1**2*L_2**2*m_2**2 + 4*J_2*L_1**2*m_1 + 4*J_1*L_2**2*m_2 + 16*J_2*L_1**2*m_2 + L_1**2*L_2**2*m_1*m_2 - 2*L_1**2*L_2**2*m_2**2*cos(2*theta_1 - 2*theta_2))
        
        theta_ddot_2 = (16*J_1*tau_2 + 16*J_1*c_2*theta_dot_1 - 16*J_1*c_2*theta_dot_2 + 16*J_1*k_2*theta_1 - 16*J_1*k_2*theta_2 + 4*L_1**2*m_1*tau_2 + 16*L_1**2*m_2*tau_2 - 4*J_1*L_2**2*c_w_2*theta_dot_2 + 4*L_1**2*c_2*m_1*theta_dot_1 - 4*L_1**2*c_2*m_1*theta_dot_2 + 16*L_1**2*c_2*m_2*theta_dot_1 - 16*L_1**2*c_2*m_2*theta_dot_2 + 4*L_1**2*k_2*m_1*theta_1 - 4*L_1**2*k_2*m_1*theta_2 + 16*L_1**2*k_2*m_2*theta_1 - 16*L_1**2*k_2*m_2*theta_2 - 4*L_1**2*L_2*g*m_2**2*sin(2*theta_1 - theta_2) + 4*L_1**2*L_2*m_2**2*x_ddot*cos(2*theta_1 - theta_2) + 8*L_1**3*L_2*m_2**2*theta_dot_1**2*sin(theta_1 - theta_2) - 4*L_1**2*L_2*m_2**2*y_ddot*sin(2*theta_1 - theta_2) + 4*L_1**2*L_2*g*m_2**2*sin(theta_2) - 4*L_1**2*L_2*m_2**2*x_ddot*cos(theta_2) + 4*L_1**2*L_2*m_2**2*y_ddot*sin(theta_2) - 8*J_1*L_2*c_w_2*x_dot*cos(theta_2) + 8*J_1*L_2*g*m_2*sin(theta_2) - 8*J_1*L_2*m_2*x_ddot*cos(theta_2) + 8*J_1*L_2*c_w_2*y_dot*sin(theta_2) + 8*J_1*L_2*m_2*y_ddot*sin(theta_2) + 2*L_1**2*L_2**2*m_2**2*theta_dot_2**2*sin(2*theta_1 - 2*theta_2) + 4*L_1*L_2*k_1*m_2*cos(theta_1 - theta_2) - 8*L_1*L_2*m_2*tau_1*cos(theta_1 - theta_2) - 8*J_1*L_2*c_w_2*w*sin(theta_2 - zeta) - L_1**2*L_2**2*c_w_2*m_1*theta_dot_2 - 4*L_1**2*L_2**2*c_w_2*m_2*theta_dot_2 + 2*L_1**2*L_2*c_w_1*m_2*x_dot*cos(2*theta_1 - theta_2) - 2*L_1**2*L_2*g*m_1*m_2*sin(2*theta_1 - theta_2) + 2*L_1**2*L_2*m_1*m_2*x_ddot*cos(2*theta_1 - theta_2) - 2*L_1**2*L_2*c_w_1*m_2*y_dot*sin(2*theta_1 - theta_2) + 2*L_1**3*L_2*m_1*m_2*theta_dot_1**2*sin(theta_1 - theta_2) - 8*J_1*L_1*L_2*c_w_2*theta_dot_1*cos(theta_1 - theta_2) - 2*L_1**2*L_2*m_1*m_2*y_ddot*sin(2*theta_1 - theta_2) - 2*L_1**2*L_2*c_w_1*m_2*w*sin(theta_2 - 2*theta_1 + zeta) + 8*L_1*L_2*c_1*m_2*theta_dot_1*cos(theta_1 - theta_2) + 8*L_1*L_2*c_2*m_2*theta_dot_1*cos(theta_1 - theta_2) - 8*L_1*L_2*c_2*m_2*theta_dot_2*cos(theta_1 - theta_2) + 8*L_1*L_2*k_2*m_2*theta_1*cos(theta_1 - theta_2) - 8*L_1*L_2*k_2*m_2*theta_2*cos(theta_1 - theta_2) + 2*L_1**2*L_2*c_w_1*m_2*x_dot*cos(theta_2) - 2*L_1**2*L_2*c_w_2*m_1*x_dot*cos(theta_2) - 8*L_1**2*L_2*c_w_2*m_2*x_dot*cos(theta_2) - 2*L_1**2*L_2*c_w_1*m_2*y_dot*sin(theta_2) + 2*L_1**2*L_2*c_w_2*m_1*y_dot*sin(theta_2) + 8*L_1**2*L_2*c_w_2*m_2*y_dot*sin(theta_2) + 8*J_1*L_1*L_2*m_2*theta_dot_1**2*sin(theta_1 - theta_2) + 2*L_1**3*L_2*c_w_1*m_2*theta_dot_1*cos(theta_1 - theta_2) - 2*L_1**3*L_2*c_w_2*m_1*theta_dot_1*cos(theta_1 - theta_2) - 8*L_1**3*L_2*c_w_2*m_2*theta_dot_1*cos(theta_1 - theta_2) + 2*L_1**2*L_2*c_w_1*m_2*w*sin(theta_2 - zeta) - 2*L_1**2*L_2*c_w_2*m_1*w*sin(theta_2 - zeta) - 8*L_1**2*L_2*c_w_2*m_2*w*sin(theta_2 - zeta))/(16*J_1*J_2 + 2*L_1**2*L_2**2*m_2**2 + 4*J_2*L_1**2*m_1 + 4*J_1*L_2**2*m_2 + 16*J_2*L_1**2*m_2 + L_1**2*L_2**2*m_1*m_2 - 2*L_1**2*L_2**2*m_2**2*cos(2*theta_1 - 2*theta_2))

        # Wind dynamics
        w_dot = 0.0
        zeta_dot = 0.0

        # Return state derivatives
        return [theta_dot_1, theta_dot_2,
                theta_ddot_1, theta_ddot_2,
                x_dot, y_dot,
                x_ddot, y_ddot,
                w_dot, zeta_dot]

    def f(self, X, U):
        """ Dynamic model for the dual antenna system.
        
        Parameters:
        -----------
        X : array-like
            Combined state vector: [X_left, X_right] where each is length 10
        U : array-like  
            Combined input vector: [U_left, U_right] where each is length 4
        """
        
        # Split the combined state and input vectors
        X_l = X[:10]  # Left antenna states
        X_r = X[10:]  # Right antenna states
        
        U_l = U[:4]   # Left antenna inputs
        U_r = U[4:]   # Right antenna inputs
        
        # Create parameter dictionaries for each antenna
        p = self.parameters
        
        params_l = {
            'L_1': p.L_1_l, 'L_2': p.L_2_l,
            'm_1': p.m_1_l, 'm_2': p.m_2_l,
            'k_1': p.k_1_l, 'k_2': p.k_2_l,
            'c_1': p.c_1_l, 'c_2': p.c_2_l,
            'J_1': p.J_1_l, 'J_2': p.J_2_l,
            'g': p.g,
            'c_w_1': p.c_w_1_l, 'c_w_2': p.c_w_2_l
        }
        
        params_r = {
            'L_1': p.L_1_r, 'L_2': p.L_2_r,
            'm_1': p.m_1_r, 'm_2': p.m_2_r,
            'k_1': p.k_1_r, 'k_2': p.k_2_r,
            'c_1': p.c_1_r, 'c_2': p.c_2_r,
            'J_1': p.J_1_r, 'J_2': p.J_2_r,
            'g': p.g,
            'c_w_1': p.c_w_1_r, 'c_w_2': p.c_w_2_r
        }
        
        # Compute dynamics for each antenna independently
        X_dot_l = self.f_single_antenna(X_l, U_l, params_l)
        X_dot_r = self.f_single_antenna(X_r, U_r, params_r)
        
        # Combine and return
        return X_dot_l + X_dot_r

    def h(self, X, U):
        """ Measurement model for dual antenna system.
        """
        
        # Split states
        X_l = X[:10]
        X_r = X[10:]
        
        # Extract angles
        theta_1_l, theta_2_l = X_l[0], X_l[1]
        theta_1_r, theta_2_r = X_r[0], X_r[1]
        
        # Compute relative angles
        theta_2_1_l = theta_2_l - theta_1_l
        theta_2_1_r = theta_2_r - theta_1_r
        
        # Return all states, inputs, and measurements
        Y = list(X) + list(U) + [theta_2_1_l, theta_2_1_r]
        return Y

    def get_left_states(self, X):
        """Extract left antenna states from combined state vector."""
        return X[:10]
    
    def get_right_states(self, X):
        """Extract right antenna states from combined state vector."""
        return X[10:]
    
    def get_left_inputs(self, U):
        """Extract left antenna inputs from combined input vector."""
        return U[:4]
    
    def get_right_inputs(self, U):
        """Extract right antenna inputs from combined input vector."""
        return U[4:]
    
    def combine_states(self, X_l, X_r):
        """Combine separate left and right antenna states into single vector."""
        return list(X_l) + list(X_r)
    
    def combine_inputs(self, U_l, U_r):
        """Combine separate left and right antenna inputs into single vector."""
        return list(U_l) + list(U_r)


# Example usage:
if __name__ == "__main__":
    # Create parameters with different values for each antenna
    params = DualAntennaParameters(
        # Left antenna - smaller and lighter
        L_1_l=0.8, L_2_l=0.6, m_1_l=0.5, m_2_l=0.3,
        # Right antenna - larger and heavier  
        L_1_r=1.2, L_2_r=1.0, m_1_r=0.8, m_2_r=0.6
    )
    
    # Create model
    model = DualAntennaModel(parameters=params)
    
    # Example initial conditions and inputs
    # Left antenna initial state
    X_l = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  
    # Right antenna initial state  
    X_r = [-0.1, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    
    # Combine states
    X = model.combine_states(X_l, X_r)
    
    # Independent inputs for each antenna
    U_l = [0.0, 0.0, 0.1, 0.05]  # Left antenna: no base accel, small torques
    U_r = [0.0, 0.0, -0.08, 0.03]  # Right antenna: no base accel, different torques
    
    # Combine inputs
    U = model.combine_inputs(U_l, U_r)
    
    # Compute dynamics
    X_dot = model.f(X, U)
    
    # Split the results to see individual antenna dynamics
    X_dot_l = model.get_left_states(X_dot)
    X_dot_r = model.get_right_states(X_dot)
    
    print("Left antenna state derivatives:")
    for i, name in enumerate(['theta_1_l', 'theta_2_l', 'theta_dot_1_l', 'theta_dot_2_l', 
                              'x_l', 'y_l', 'x_dot_l', 'y_dot_l', 'w_l', 'zeta_l']):
        print(f"  {name}: {X_dot_l[i]:.6f}")
    
    print("\nRight antenna state derivatives:")
    for i, name in enumerate(['theta_1_r', 'theta_2_r', 'theta_dot_1_r', 'theta_dot_2_r',
                              'x_r', 'y_r', 'x_dot_r', 'y_dot_r', 'w_r', 'zeta_r']):
        print(f"  {name}: {X_dot_r[i]:.6f}")
    
    # Compute measurements
    Y = model.h(X, U)
    print(f"\nMeasurements:")
    print(f"  Relative angle left antenna (theta_2_1_l): {Y[-2]:.6f}")
    print(f"  Relative angle right antenna (theta_2_1_r): {Y[-1]:.6f}")