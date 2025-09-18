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