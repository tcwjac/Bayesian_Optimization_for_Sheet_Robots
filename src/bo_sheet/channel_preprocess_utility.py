import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, LineString
from scipy.interpolate import PchipInterpolator, CubicSpline, make_splprep, interp1d, splprep, splev
from enum import Enum
import tkinter as tk
from bo_sheet.curve_compute_utils import *

class ChannelType(Enum):
    """
    Previously all four channel types are in code comments format. They are formatted to Enum class
    for convenience.
    """
    simple_straight = 'simple_straight'
    arc = 'arc'
    straight = 'straight'
    s_shaped = 's_shaped'
    sinusoidal = 'sinusoidal'
    arbitrary = 'arbitrary'

def plot_channel_all_info(x_midline: npt.NDArray[np.float64],
                          y_midline: npt.NDArray[np.float64],
                          channel_polygon: Polygon,
                          left_bank: npt.NDArray[np.float64],
                          right_bank: npt.NDArray[np.float64]):
    """
    Plot the channel midline and channel polygon on the left panel and plot the left and right banks on the right panel.
    """
    # Create two subplots side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1: Axes
    ax2: Axes

    # Left panel: Plot midline and channel polygon.
    ax1.plot(x_midline, y_midline, 'b-', lw=1.5, label="Midline")
    x_poly, y_poly = channel_polygon.exterior.xy
    ax1.plot(x_poly, y_poly, 'r--', lw=1, label="Channel Boundary")
    ax1.fill(x_poly, y_poly, color='lightblue', alpha=0.5)
    ax1.set_title("Channel Profile")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.legend()

    # Right panel: Plot left and right banks.
    left_x, left_y = left_bank
    right_x, right_y = right_bank
    ax2.plot(left_x, left_y, 'g-', lw=1.5, label="Left Bank")
    ax2.plot(right_x, right_y, 'm-', lw=1.5, label="Right Bank")
    ax2.set_title("Channel Banks")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.axis("equal")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def midline_construction(
    channel_type: str,
    total_channel_length: float,
    N_points: float,
    amplitude: float,
    frequency: float,
    radius: float,
    origin_offset: float,
    backward: bool,
    init_wall_to_closest_tip_margin=0.0,
    insertion_margin=0.0,
    curved_section=0.0,
    ending_margin=0.0
):
    """
    Returns
    -------
        x_midline (In backward format if backward is True)

        y_midline (In backward format if backward is True)

        unit_normal_vector (based on signed curvature. In backward format if backward is True) for width construction

        total_channel_length (Negative if backward is True)
    """
    if backward == True:
        total_channel_length = -total_channel_length

    s_increments: npt.NDArray[np.float64] = np.linspace(0, total_channel_length, N_points)

    if channel_type == 's_shaped':
        x_midline: npt.NDArray[np.float64] = s_increments
        y_midline: npt.NDArray[np.float64] = amplitude * np.tanh(4 * (x_midline / total_channel_length - 0.5)) * np.sin(np.pi * x_midline / total_channel_length)

    elif channel_type == 'sinusoidal':
        x_midline = s_increments
        start_percent_curved_section = (init_wall_to_closest_tip_margin + insertion_margin) / total_channel_length
        end_percent_curved_section = (init_wall_to_closest_tip_margin + insertion_margin + curved_section) / total_channel_length
        start_index_curved = int(N_points * start_percent_curved_section)
        end_index_curved = int(N_points * end_percent_curved_section)

        y_midline = np.zeros_like(x_midline)

        # Start at sin(pi/2) and end at sin(pi/2) to ensure smoothness
        # The start and end y-value should match y=0
        normalized_curved_x = np.linspace(0, 1, end_index_curved - start_index_curved)
        y_midline[start_index_curved:end_index_curved] = amplitude * np.sin(2 * np.pi * frequency * normalized_curved_x + (np.pi / 2.0)) - amplitude
        
    elif channel_type == 'arc':
        # Circular arc channel
        angle_total = curved_section / radius

        if np.abs(angle_total) >= 2*np.pi:
            raise ValueError(f"Invalid circular arc channel. L_total is >= 2pi.")

        # Calculate the number of points for the arc section
        angles = np.linspace(0, angle_total, N_points)
        
        # Generate the arc
        x_arc = radius * np.sin(angles)
        y_arc = radius * (1 - np.cos(angles))
        
        # Calculate tangent directions at start and end
        start_angle = 0
        end_angle = angle_total
        
        # Tangent vectors normalized
        start_tangent = np.array([np.cos(start_angle), np.sin(start_angle)])
        end_tangent = np.array([np.cos(end_angle), np.sin(end_angle)])
        
        # Extend the start, backwards along the tangent
        start_ext_length = init_wall_to_closest_tip_margin + insertion_margin
        start_ext_param = np.linspace(-start_ext_length, 0, N_points)
        start_ext_param = start_ext_param[:-1]
        x_start_ext = x_arc[0] + start_ext_param * start_tangent[0]
        y_start_ext = y_arc[0] + start_ext_param * start_tangent[1]
        
        # Extend the end, forwards along the tangent
        end_ext_length = ending_margin
        end_ext_param = np.linspace(0, end_ext_length, N_points + 1)[1:]  # Skip the first point (already in arc)
        x_end_ext = x_arc[-1] + end_ext_param * end_tangent[0]
        y_end_ext = y_arc[-1] + end_ext_param * end_tangent[1]
        
        # Combine all segments
        x_midline = np.concatenate([x_start_ext, x_arc, x_end_ext])
        y_midline = np.concatenate([y_start_ext, y_arc, y_end_ext])

    midline = LineString(np.column_stack((x_midline, y_midline)))

    # Since u argument isn't provided in make_splprep, u_param is always normalized to [0, 1] regardless of
    # the negative trend of x_midline.
    spline, _ = make_splprep([x_midline, y_midline])
    u_param = np.linspace(0, 1, x_midline.shape[0])

    # Shape (xy, N_points)
    dr_du = spline.derivative()
    dr_du_value = dr_du(u_param)

    # Shape (N_points, xy)
    dr_du_value = dr_du_value.T
    norm_dr_du = np.linalg.norm(dr_du_value, axis=1)

    # Final check on the curve
    if not midline.is_simple:
        plt.figure(figsize=(10, 5))

        plt.plot(x_midline, y_midline, 
                color='red', 
                linewidth=2, 
                label='Midline')

        plt.xlabel('Cartesian x-axis', fontsize=12)
        plt.ylabel('Cartesian y-axis', fontsize=12)
        plt.title('Problematic midline', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        raise ValueError(f"Midline has self-intersection.")

    # Translate the channel so that (0, 0) lies along the midline at the specified offset
    target_index = int(origin_offset * (N_points - 1))
    x_translate: float = -x_midline[target_index]
    y_translate: float = -y_midline[target_index]
    
    # Apply translation to the midline
    # Shape (N_points,)
    x_midline = x_midline + x_translate
    y_midline = y_midline + y_translate

    # Shape (N_points, xy)
    unit_tangent_vector: npt.NDArray[np.float64] = dr_du_value / norm_dr_du[:, np.newaxis]

    # N(u) = [-T_y, T_x] for 2D because we're finding the outward unit normal of curve
    # for left and right bank construction later.
    unit_normal_vector: npt.NDArray[np.float64] = np.stack((-unit_tangent_vector[:, 1], unit_tangent_vector[:, 0]), axis=1)

    return x_midline, y_midline, unit_normal_vector, total_channel_length

def width_construction(
    channel_type: str,
    total_channel_length: float,
    num_width_ctrl: int,
    min_width: float,
    max_width: float,
    max_rate_width: float,
    x_midline: npt.NDArray[np.float64],
    y_midline: npt.NDArray[np.float64],
    unit_normal_vector: npt.NDArray[np.float64],
    width: float,
    constant_width: bool,
    init_wall_to_closest_tip_margin=0.0,
    insertion_margin=0.0,
    curved_section=0.0,
    ending_margin=0.0
):
    """
    channel_polygon must be a closed polygon along one direction, hence the addition of last and first points from\\
    right_points and left_points respectively, and reversed right bank.\n

    left_bank and right_bank however doesn't follow this rule.\\
    They simply follow along midline's direction.

    The returned left_bank and right bank are along midline's direction.\n
    We don't return the reversed right bank.
    """
    if not constant_width:
        dist_from_curve_start = init_wall_to_closest_tip_margin + insertion_margin
        s_width_ctrl: npt.NDArray[np.float64] = np.linspace(0.0, total_channel_length, num_width_ctrl)
        width_ctrl: npt.NDArray[np.float64] = np.random.uniform(min_width, max_width, num_width_ctrl)

        # Hack for sinusoidal.
        # width_ctrl = np.array([1.8, 0.6, 3.0, 1.8])
        # s_width_ctrl: npt.NDArray[np.float64] = np.array([0.0, curved_section / 3.0, 2 * curved_section / 3.0, curved_section])

        # Enforce maximum width rate.
        for i in range(s_width_ctrl.shape[0] - 2):
            ds_width: np.float64 = s_width_ctrl[i + 1] - s_width_ctrl[i]
            diff_w: np.float64 = width_ctrl[i + 1] - width_ctrl[i]
            rate_w = np.abs(diff_w) / ds_width
            if rate_w > max_rate_width:
                width_ctrl[i + 1] = width_ctrl[i] + np.sign(diff_w) * max_rate_width * ds_width

        width_spline = PchipInterpolator(s_width_ctrl, width_ctrl)

        s_discretized = np.linspace(0.0, dist_from_curve_start + curved_section + ending_margin, x_midline.shape[0])
        width_vals = width_spline(s_discretized)
    else:
        width_vals = width

    # Compute the left and right banks
    n_x = unit_normal_vector[:, 0]
    n_y = unit_normal_vector[:, 1]
    half_width = 0.5 * width_vals
    left_x = x_midline + half_width * n_x
    left_y = y_midline + half_width * n_y
    right_x = x_midline - half_width * n_x
    right_y = y_midline - half_width * n_y

    left_points = list(zip(left_x, left_y))
    right_points = list(zip(right_x, right_y))

    right_points_rev = right_points[::-1]
    polygon_points = left_points + [right_points[-1]] + right_points_rev + [left_points[0]]
    channel_polygon = Polygon(polygon_points)

    left_bank: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (left_x, left_y)
    right_bank: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (right_x, right_y)

    # Check that the polygon is valid.
    if not channel_polygon.is_valid:
        plot_channel_all_info(x_midline, y_midline, channel_polygon, left_bank, right_bank)
        raise ValueError(f"Generated {channel_type} type channel polygon is invalid.")
    
    return x_midline, y_midline, channel_polygon, left_bank, right_bank

def generate_closed_channel(total_channel_length: float,
                            n_bending: int,
                            num_width_ctrl: int,
                            max_curvature: float,
                            min_curvature: float,
                            max_width: float,
                            min_width: float,
                            max_rate_curvature: float,
                            max_rate_width: float,
                            channel_type='simple_straight',
                            width=2.0,
                            amplitude: float | None=None,
                            frequency: float | None=None,
                            radius: float | None=None,
                            constant_width=True,
                            origin_offset=0.0,
                            N_points=1000,
                            backward=False,
                            init_wall_to_closest_tip_margin=0.0,
                            insertion_margin=0.0,
                            curved_section=0.0,
                            ending_margin=0.0):
    """
    Generate a 2D channel profile with a specified number of bending points and closed at both ends.

    Parameters:
        total_channel_length : float
            Total length of the channel midline.
        n_bending : int
            Number of internal bending (Arc directional change of curve).
            This value is ignored if channel_type is simple_straight.
        num_width_ctrl
            Number of channel width random variations.
            This value is ignored if channel_type is simple_straight.
        max_curvature : float
            Maximum curvature (absolute value) for positive bending.
            This value is ignored if channel_type is simple_straight.
        min_curvature : float
            Minimum curvature (absolute value) for negative bending.
            This value is ignored if channel_type is simple_straight.
        max_width : float
            Maximum channel width.
            This value is ignored if channel_type is simple_straight.
        min_width : float
            Minimum channel width.
            This value is ignored if channel_type is simple_straight.
        max_rate_curvature : float
            Maximum allowed rate of change of curvature (|dk/ds|).
            This value is ignored if channel_type is simple_straight.
        max_rate_width : float
            Maximum allowed rate of change of width (|dw/ds|).
            This value is ignored if channel_type is simple_straight.
        channel_type : str, optional
            Type of channel to generate. Options: 'simple_straight', 'straight', 's_shaped', 'sinusoidal', 'arc', 'arbitrary'.
            Default is 'simple_straight'.
        width : float, optional
            Width of the channel for constant_wdith True. Default is 2.0.
        amplitude : float, optional
            Amplitude for sinusoidal and s_shaped channels. If None, defaults to L_total/10.
        frequency : float, optional
            Frequency for sinusoidal channels (number of cycles). If None, defaults to 2.
        radius : float, optional
            Radius for arc channels. If None, defaults to L_total/np.pi (semicircle).
        origin_offset : float, optional
            Position along the midline where (0, 0) should be placed. Range: 0.0 to 1.0 (0% to 100%).
            0.0 means (0, 0) at the starting point, 1.0 means (0, 0) at the ending point.
            No meaning for arc.
        N_points : int, optional
            Number of discretization points for curved channels. Default is 1000.
            Note: simple_straight channel use only 4 vertices regardless of this parameter.
            1 / N_points should ideally output a rational number.
        backward: bool, optional
            Set up the channel for backward crawling case if True. DOESN'T SUPPORT ARBITRARY.

    Returns:
        x_midline (In backward format if backward is True) : numpy.ndarray
            1D array of midline x-coordinates.
        y_midline (In backward format if backward is True) : numpy.ndarray
            1D array of midline y-coordinates.
        channel_polygon : shapely.geometry.Polygon
            Closed polygon representing the channel.
        left_bank (In backward format if backward is True, i.e., left from the perspective of the x_midline trend) : tuple of two numpy.ndarray
            Tuple (left_x, left_y) representing the left bank coordinates.
        right_bank (In backward format if backward is True, i.e., right from the perspective of the x_midline trend): tuple of two numpy.ndarray
            Tuple (right_x, right_y) representing the right bank coordinates.
    """

    if amplitude is None:
        amplitude = total_channel_length / 10.0
    if frequency is None:
        frequency = 2.0
    if radius is None:
        radius = total_channel_length / np.pi

    if not (0.0 <= origin_offset <= 1.0):
        raise ValueError(f"origin_offset must be between 0.0 and 1.0, got {origin_offset}")
    if N_points < 10:
        raise ValueError(f"N_points must be at least 10, got {N_points}")

    # Handle straight channel separately for efficiency
    if channel_type == 'simple_straight':
        # Create simple rectangular channel with just 4 vertices
        half_width = 0.5 * width
        half_length = 0.5 * total_channel_length
        
        # Position channel so (0, 0) is at the desired offset along the midline
        x_center = -origin_offset * total_channel_length + half_length
        
        # Create simple rectangle polygon
        channel_polygon = Polygon([
            (x_center - half_length, -half_width),
            (x_center + half_length, -half_width),
            (x_center + half_length, half_width),
            (x_center - half_length, half_width)
        ])
        
        # Create simple midline (just for compatibility)
        x_midline: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
        y_midline: npt.NDArray[np.float64] = np.array([0.0, 0.0])
        
        # Create simple banks
        left_x: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
        left_y: npt.NDArray[np.float64] = np.array([half_width, half_width])
        right_x: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
        right_y: npt.NDArray[np.float64] = np.array([-half_width, -half_width])
        
        if backward == True:
            x_midline = x_midline[::-1]
            y_midline = y_midline[::-1]
            left_x = left_x[::-1]
            left_y = left_y[::-1]
            right_x = right_x[::-1]
            right_y = right_y[::-1]

        left_bank = (left_x, left_y)
        right_bank = (right_x, right_y)
        
        return x_midline, y_midline, channel_polygon, left_bank, right_bank

    if (N_points / total_channel_length) < 10:
        N_points *= 10

    if channel_type == 'straight' or channel_type == 'arc' or channel_type == 's_shaped' or channel_type == 'sinusoidal':
        if channel_type != 'straight':
            x_midline, y_midline, unit_normal_vector, total_channel_length = midline_construction(
                channel_type, total_channel_length, N_points, amplitude, frequency, radius,
                origin_offset, backward, init_wall_to_closest_tip_margin, insertion_margin,
                curved_section, ending_margin)
            
            x_midline, y_midline, channel_polygon, left_bank, right_bank = width_construction(
                channel_type, total_channel_length, num_width_ctrl, min_width, max_width, max_rate_width,
                x_midline, y_midline, unit_normal_vector, width, constant_width,
                init_wall_to_closest_tip_margin, insertion_margin, curved_section, ending_margin
            )

            return x_midline, y_midline, channel_polygon, left_bank, right_bank
            
        else:
            if backward == True:
                total_channel_length = -total_channel_length
                
            s_increments: npt.NDArray[np.float64] = np.linspace(0, total_channel_length, N_points)

            x_midline: npt.NDArray[np.float64] = np.array(s_increments)
            y_midline: npt.NDArray[np.float64] = np.zeros_like(x_midline)
            unit_normal_vector: npt.NDArray[np.float64] = np.stack((np.zeros_like(x_midline), np.ones_like(x_midline)), axis=1)

            # Translate the channel so that (0, 0) lies along the midline at the specified offset
            target_index = int(origin_offset * (N_points - 1))
            x_translate: float = -x_midline[target_index]
            y_translate: float = -y_midline[target_index]
            
            # Apply translation to the midline
            x_midline = x_midline + x_translate
            y_midline = y_midline + y_translate

            x_midline, y_midline, channel_polygon, left_bank, right_bank = width_construction(
                channel_type, total_channel_length, num_width_ctrl, min_width, max_width, max_rate_width,
                x_midline, y_midline, unit_normal_vector, width, constant_width
            )

            return x_midline, y_midline, channel_polygon, left_bank, right_bank

    elif channel_type == 'arbitrary':
        # Define a margin to avoid placing bending points too near the ends.
        margin = 12.0
        # Generate n_bending internal s values uniformly between margin and L_total - margin.
        s_internal = np.sort(np.random.uniform(margin, total_channel_length - margin, n_bending))
        # Control points: endpoints at 0 and L_total.
        s_ctrl: npt.NDArray[np.float64] = np.concatenate(([0], s_internal, [total_channel_length]))

        # Set curvature values at control points.
        # Endpoints: 0 curvature.
        # For internal points, alternate sign: even indices get positive, odd get negative.
        k_ctrl: List = [0]
        for i in range(n_bending):
            if i % 2 == 0:
                k_ctrl.append(np.random.uniform(0, max_curvature))
            else:
                k_ctrl.append(-np.random.uniform(0, min_curvature))
        k_ctrl.append(0)
        k_ctrl = np.array(k_ctrl)

        # Enforce maximum rate of change of curvature
        for i in range(len(s_ctrl) - 1):
            ds = s_ctrl[i + 1] - s_ctrl[i]
            diff = k_ctrl[i + 1] - k_ctrl[i]
            rate = np.abs(diff) / ds
            if rate > max_rate_curvature:
                # Saturate the difference.
                k_ctrl[i + 1] = k_ctrl[i] + np.sign(diff) * max_rate_curvature * ds

        curvature_spline = CubicSpline(s_ctrl, k_ctrl)

        # Discretize the midline.
        s_vals: npt.NDArray[np.float64] = np.linspace(0, total_channel_length, N_points)
        ds: np.float64 = s_vals[1] - s_vals[0]

        # Interpolated curvature values given arc points with the interpolation function generated from s_ctrl and k_ctrl
        k_vals = curvature_spline(s_vals)
        theta_vals = np.cumsum(k_vals) * ds

        # Project arc onto Cartesian Coordinate
        x_midline: npt.NDArray[np.float64] = np.cumsum(np.cos(theta_vals)) * ds
        y_midline: npt.NDArray[np.float64] = np.cumsum(np.sin(theta_vals)) * ds

        # Check if the midline is self-intersecting.
        midline = LineString(np.column_stack((x_midline, y_midline)))
        if not midline.is_simple:
            plt.figure(figsize=(10, 5))
            plt.plot(x_midline, y_midline, 
                    color='red', 
                    linewidth=2, 
                    label='Midline')
            plt.xlabel('Cartesian x-axis', fontsize=12)
            plt.ylabel('Cartesian y-axis', fontsize=12)
            plt.title('Problematic midline', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
            raise ValueError("Midline has self-intersection.")

        # Generate a smooth width function
        s_width_ctrl: npt.NDArray[np.float64] = np.linspace(0, total_channel_length, num_width_ctrl)
        # Shape (num_width_ctrl,)
        width_ctrl: npt.NDArray[np.float64] = np.random.uniform(min_width, max_width, num_width_ctrl)
        # Enforce maximum width rate.
        for i in range(num_width_ctrl - 1):
            ds_width: np.float64 = s_width_ctrl[i + 1] - s_width_ctrl[i]
            diff_w: np.float64 = width_ctrl[i + 1] - width_ctrl[i]
            rate_w = np.abs(diff_w) / ds_width
            if rate_w > max_rate_width:
                width_ctrl[i + 1] = width_ctrl[i] + np.sign(diff_w) * max_rate_width * ds_width

        width_spline = CubicSpline(s_width_ctrl, width_ctrl)
        width_vals = width_spline(s_vals)

        # Compute the left and right banks.
        n_x = -np.sin(theta_vals)
        n_y = np.cos(theta_vals)
        half_width = 0.5 * width_vals
        left_x = x_midline + half_width * n_x
        left_y = y_midline + half_width * n_y
        right_x = x_midline - half_width * n_x
        right_y = y_midline - half_width * n_y

        left_points = list(zip(left_x, left_y))
        right_points = list(zip(right_x, right_y))

        right_points_rev = right_points[::-1]
        polygon_points = left_points + [right_points[-1]] + right_points_rev + [left_points[0]]
        channel_polygon = Polygon(polygon_points)

        # Check that the polygon is valid.
        if not channel_polygon.is_valid:
            plot_channel_all_info(x_midline, y_midline, channel_polygon, left_bank, right_bank)
            raise ValueError(f"Generated {channel_type} type channel polygon is invalid.")

        left_bank: npt.NDArray[np.float64] = (left_x, left_y)
        right_bank: npt.NDArray[np.float64] = (right_x, right_y)
        return x_midline, y_midline, channel_polygon, left_bank, right_bank
    
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}.\n"
                        f"Valid options are: 'simple_straight', 'straight', 's_shaped', 'sinusoidal', 'arc', 'arbitary'")

def create_smooth_curve(
    points_array: npt.NDArray[np.float64],
    N_points: int,
    s=0
) -> npt.NDArray[np.float64]:
    """
    Inputs:
        points_array of shape (num of points, x and y)
        N_points how many points to be used in smoothing

    Outputs:
        xy_smooth_curve of shape (N_points, x and y)
    """
    tck, _ = splprep([points_array[:, 0], points_array[:, 1]], s=s)
    spline_x, spline_y = splev(np.linspace(0, 1, N_points), tck)

    xy_smooth_curve = np.column_stack((spline_x, spline_y))
    return xy_smooth_curve

def find_closest_idx_on_boundary(
    single_point: bool,
    midline: npt.NDArray[np.float64],
    midline_normal_vectors: npt.NDArray[np.float64],
    multiplying_factor: npt.NDArray[np.float64],
    left_boundary: npt.NDArray[np.float64],
    right_boundary: npt.NDArray[np.float64]
) -> Tuple[List[int], List[int]]:
    """
    The midline doesn't necessary need to be shape (N_points, xy). You can also provide shape (1, xy), i.e., a single point.
    Providing a single midline point is used in simplifying the start and end left right boundary to match where the midline is.
    Please supply True to single_point argument, and the function returns two int.

    Otherwise, this function default's behavior is to find the index on the left and right boundary closest to the midline normal vector
    casted line.

    Return two List in total, each List for indexing the left or right boundary that's closest to the i-th index of the midline

    Returns
        (left_idx_list, right_idx_list)
    """

    right_idx_list = []
    left_idx_list = []
    left_tree = KDTree(left_boundary)
    right_tree = KDTree(right_boundary)

    # Singular point case
    if single_point:
        perpendicular_casted_line = midline + midline_normal_vectors * multiplying_factor[:, np.newaxis]
        
        left_dists, left_indices = left_tree.query(perpendicular_casted_line)
        right_dists, right_indices = right_tree.query(perpendicular_casted_line)
        
        left_min_idx = np.argmin(left_dists)
        right_min_idx = np.argmin(right_dists)
        
        left_closest_point_idx: int = left_indices[left_min_idx]
        right_closest_point_idx: int = right_indices[right_min_idx]

        return left_closest_point_idx, right_closest_point_idx
    
    else:
        for i in range(midline.shape[0]):
            mid_point = midline[i, :]

            perpendicular_casted_line = mid_point + midline_normal_vectors[i, :] * multiplying_factor[:, np.newaxis]
            
            left_dists, left_indices = left_tree.query(perpendicular_casted_line)
            right_dists, right_indices = right_tree.query(perpendicular_casted_line)
            
            left_min_idx = np.argmin(left_dists)
            right_min_idx = np.argmin(right_dists)
            
            left_closest_point_idx = left_indices[left_min_idx]
            right_closest_point_idx = right_indices[right_min_idx]

            right_idx_list.append(right_closest_point_idx)
            left_idx_list.append(left_closest_point_idx)

        return left_idx_list, right_idx_list

def create_centered_figure(
    figsize=(8,8)
):
    """
    matplotlib.pyplot is quite annoying that it doesn't center the figure
    to the center of the screen
    """
    if not hasattr(create_centered_figure, 'screen_width'):
        root = tk.Tk()
        create_centered_figure.screen_width = root.winfo_screenwidth()
        create_centered_figure.screen_height = root.winfo_screenheight()
        root.destroy()
    
    fig = plt.figure(figsize=figsize)
    
    dpi = fig.get_dpi()
    width_pixels = figsize[0] * dpi
    height_pixels = figsize[1] * dpi
    x_pos = int((create_centered_figure.screen_width - width_pixels) // 2)
    y_pos = int((create_centered_figure.screen_height - height_pixels) // 2)
    
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry(f"+{x_pos}+{y_pos}")
    
    return fig

def segment_channel_extreme(x_midline: npt.NDArray[np.float64],
                            y_midline: npt.NDArray[np.float64],
                            left_bank: npt.NDArray[np.float64],
                            right_bank: npt.NDArray[np.float64],
                            L_window: float,
                            N_points: int):
    """
    Intended for Arbitrary.
    Slide a window along the channel midline and identify 5 extreme segments:

      (1) max_curvature: the window (of length = L_window) with the largest average curvature.
      (2) max_width: the window with the largest average width.
      (3) min_width: the window with the smallest average width.
      (4) max_curvature_change: the L_window segment centered on the point where the average
          curvature change (computed over a window of length 0.1*L_window) is maximal.
      (5) max_width_change: the L_window segment centered on the point where the average
          width change (computed over a window of length 0.1*L_window) is maximal.

    Parameters:
       x_midline: 1D ndarray of midline x-coordinates.
       y_midline: 1D ndarray of midline y-coordinates.
       left_bank: tuple of two ndarrays (left_x, left_y) for left bank coordinates.
       right_bank: tuple of two ndarrays (right_x, right_y) for right bank coordinates.
       L_window: float, the desired arc-length of each extracted segment.

    Returns:
       extreme_segments: dict with keys 'max_curvature','max_width','min_width',
                         'max_curvature_change','max_width_change'.
           Each value is a tuple (seg_midline, seg_polygon), where:
              seg_midline: an array shape (100, xy) of midline points in that window.
              seg_polygon: a Polygon built by joining the corresponding left bank
                           segment and the reversed right bank segment.
    """
    extreme_segments = {}

    # Shape (N_points, xy)
    midline: npt.NDArray[np.float64] = np.column_stack((x_midline, y_midline))
    left_bank_arr: npt.NDArray[np.float64] = np.column_stack((left_bank[0], left_bank[1]))
    right_bank_arr: npt.NDArray[np.float64] = np.column_stack((right_bank[0], right_bank[1]))

    # Compute arc-length along the midline.
    arc_cumsum_lengths = compute_arc_length(midline)
    N_points = arc_cumsum_lengths.shape[0]

    # Compute width and curvature.
    width_list, right_idx_list, left_idx_list, min_width, max_width, min_indices_in_width_list, max_indices_in_width_list, min_indices_in_smooth_right_boundary, min_indices_in_smooth_left_boundary, max_indices_in_smooth_right_boundary, max_indices_in_smooth_left_boundary = compute_width(midline, left_bank_arr, right_bank_arr)
    width = np.array(width_list)
    curvature = compute_curvature(midline, arc_cumsum_lengths)

    # Compute derivatives.
    # Shape (N_points,)
    curvature_deriv: npt.NDArray[np.float64] = np.abs(np.gradient(curvature, arc_cumsum_lengths))
    width_deriv: npt.NDArray[np.float64] = np.abs(np.gradient(width, arc_cumsum_lengths))

    # NOTE: If there exists multiple widths being the same min or max value, you must pick one out.
    midline_min_width_start_idx, midline_min_width_end_idx = get_centered_window_indices(arc_cumsum_lengths, min_indices_in_width_list[0], L_window)
    midline_max_width_start_idx, midline_max_width_end_idx = get_centered_window_indices(arc_cumsum_lengths, max_indices_in_width_list[0], L_window)

    # Use the midline min or max width start end idx to slice right and left idx list
    # Then use the sliced right and left idx list on the left and right bank array to get correct section.
    min_width_right_start_idx = right_idx_list[midline_min_width_start_idx]
    min_width_right_end_idx = right_idx_list[midline_min_width_end_idx]
    min_width_left_start_idx = left_idx_list[midline_min_width_start_idx]
    min_width_left_end_idx = left_idx_list[midline_min_width_end_idx]

    max_width_right_start_idx = right_idx_list[midline_max_width_start_idx]
    max_width_right_end_idx = right_idx_list[midline_max_width_end_idx]
    max_width_left_start_idx = left_idx_list[midline_max_width_start_idx]
    max_width_left_end_idx = left_idx_list[midline_max_width_end_idx]

    min_width_left_bank = left_bank_arr[min_width_left_start_idx:min_width_left_end_idx+1, :]
    min_width_right_bank = right_bank_arr[min_width_right_start_idx:min_width_right_end_idx+1, :]

    max_width_left_bank = left_bank_arr[max_width_left_start_idx:max_width_left_end_idx+1, :]
    max_width_right_bank = right_bank_arr[max_width_right_start_idx:max_width_right_end_idx+1, :]

    min_width_midline = midline[midline_min_width_start_idx:midline_min_width_end_idx+1, :]
    max_width_midline = midline[midline_max_width_start_idx:midline_max_width_end_idx+1, :]

    seg_mid_min_width = create_smooth_curve(min_width_midline, N_points)
    seg_mid_max_width = create_smooth_curve(max_width_midline, N_points)

    seg_right_min_width = create_smooth_curve(min_width_right_bank, N_points)
    seg_right_max_width = create_smooth_curve(max_width_right_bank, N_points)

    seg_left_min_width = create_smooth_curve(min_width_left_bank, N_points)
    seg_left_max_width = create_smooth_curve(max_width_left_bank, N_points)

    seg_coords = np.vstack((seg_left_min_width, seg_right_min_width[::-1, :]))
    seg_poly = Polygon(seg_coords)
    extreme_segments['min_width'] = (seg_poly, seg_mid_min_width, seg_left_min_width, seg_right_min_width, L_window)

    seg_coords = np.vstack((seg_left_max_width, seg_right_max_width[::-1, :]))
    seg_poly = Polygon(seg_coords)
    extreme_segments['max_width'] = (seg_poly, seg_mid_max_width, seg_left_max_width, seg_right_max_width, L_window)

    avg_curv_metrics = []
    for i in range(N_points):
        # j is the ending index for the length from arc[i] length + L_window length in the arc array
        j: int = np.searchsorted(arc_cumsum_lengths, arc_cumsum_lengths[i] + L_window)

        if j < N_points:
            avg_curv: np.float64 = np.mean(curvature[i:j])
            avg_curv_metrics.append({'i': i, 'j': j, 'avg_curv': avg_curv})

    avg_curv_metrics: List[Dict[str, int | np.float64]]

    # Returned variable is a Dict of the i, j, avg_curv. i and j are for midline.
    seg_max_curv = max(avg_curv_metrics, key=lambda d: d['avg_curv'])

    max_curv_mid_start_idx = seg_max_curv['i']
    max_curv_mid_end_idx = seg_max_curv['j']

    max_curv_right_start_idx = right_idx_list[max_curv_mid_start_idx]
    max_curv_right_end_idx = right_idx_list[max_curv_mid_end_idx]

    max_curv_left_start_idx = left_idx_list[max_curv_mid_start_idx]
    max_curv_left_end_idx = left_idx_list[max_curv_mid_end_idx]

    max_curv_mid = midline[max_curv_mid_start_idx:max_curv_mid_end_idx+1, :]
    max_curv_right = right_bank_arr[max_curv_right_start_idx:max_curv_right_end_idx+1, :]
    max_curv_left = left_bank_arr[max_curv_left_start_idx:max_curv_left_end_idx+1, :]

    seg_max_curv_mid = create_smooth_curve(max_curv_mid, N_points)
    seg_max_curv_right = create_smooth_curve(max_curv_right, N_points)
    seg_max_curv_left = create_smooth_curve(max_curv_left, N_points)

    seg_coords = np.vstack((seg_max_curv_left, seg_max_curv_right[::-1, :]))
    seg_poly = Polygon(seg_coords)
    extreme_segments['max_curvature'] = (seg_poly, seg_max_curv_mid, seg_max_curv_left, seg_max_curv_right, L_window)

    # For local changes, use a window length L_local = 0.1 * L_window.
    L_local = 0.1 * L_window

    # Compute local average curvature change.
    local_curv_change = np.zeros(N_points)

    for i in range(N_points):
        i_start_local, i_end_local = get_centered_window_indices(arc_cumsum_lengths, i, L_local)
        if i_end_local > i_start_local:
            local_curv_change[i] = np.mean(curvature_deriv[i_start_local:i_end_local])
        else:
            local_curv_change[i] = 0.0

    center_idx_curv: int = np.argmax(local_curv_change)

    # Extract a segment of length L_window centered at center_idx_curv.
    i_start_curv, i_end_curv = get_centered_window_indices(arc_cumsum_lengths, center_idx_curv, L_window)

    max_curv_change_right_start_idx = right_idx_list[i_start_curv]
    max_curv_change_right_end_idx = right_idx_list[i_end_curv]

    max_curv_change_left_start_idx = left_idx_list[i_start_curv]
    max_curv_change_left_end_idx = left_idx_list[i_end_curv]

    max_curv_change_mid = midline[i_start_curv:i_end_curv+1, :]
    max_curv_change_right = right_bank_arr[max_curv_change_right_start_idx:max_curv_change_right_end_idx+1, :]
    max_curv_change_left = left_bank_arr[max_curv_change_left_start_idx:max_curv_change_left_end_idx+1, :]

    seg_max_curv_change_mid = create_smooth_curve(max_curv_change_mid, N_points)
    seg_max_curv_change_right = create_smooth_curve(max_curv_change_right, N_points)
    seg_max_curv_change_left = create_smooth_curve(max_curv_change_left, N_points)

    seg_coords = np.vstack((seg_max_curv_change_left, seg_max_curv_change_right[::-1, :]))
    seg_poly = Polygon(seg_coords)
    extreme_segments['max_curvature_change'] = (seg_poly, seg_max_curv_change_mid, seg_max_curv_change_left, seg_max_curv_change_right, L_window)

    # Compute local average width change.
    local_width_change = np.zeros(N_points)

    for i in range(N_points):
        i_start_local, i_end_local = get_centered_window_indices(arc_cumsum_lengths, i, L_local)
        if i_end_local > i_start_local:
            local_width_change[i] = np.mean(width_deriv[i_start_local:i_end_local])
        else:
            local_width_change[i] = 0.0

    center_idx_width: int = np.argmax(local_width_change)

    i_start_width, i_end_width = get_centered_window_indices(arc_cumsum_lengths, center_idx_width, L_window)

    max_width_change_right_start_idx = right_idx_list[i_start_width]
    max_width_change_right_end_idx = right_idx_list[i_end_width]

    max_width_change_left_start_idx = left_idx_list[i_start_width]
    max_width_change_left_end_idx = left_idx_list[i_end_width]

    max_width_change_mid = midline[i_start_width:i_end_width+1, :]
    max_width_change_right = right_bank_arr[max_width_change_right_start_idx:max_width_change_right_end_idx+1, :]
    max_width_change_left = left_bank_arr[max_width_change_left_start_idx:max_width_change_left_end_idx+1, :]

    seg_max_width_change_mid = create_smooth_curve(max_width_change_mid, N_points)
    seg_max_width_change_right = create_smooth_curve(max_width_change_right, N_points)
    seg_max_width_change_left = create_smooth_curve(max_width_change_left, N_points)

    seg_coords = np.vstack((seg_max_width_change_left, seg_max_width_change_right[::-1, :]))
    seg_poly = Polygon(seg_coords)
    extreme_segments['max_width_change'] = (seg_poly, seg_max_width_change_mid, seg_max_width_change_left, seg_max_width_change_right, L_window)

    extreme_segments: Dict[str, Tuple[Polygon, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]]

    return extreme_segments

def plot_extreme_segments(extreme_segments: Dict[str, Tuple[npt.NDArray[np.float64], Polygon]],
                          channel_polygon: Polygon):
    """
    Intended for Arbitrary.
    First, plot each segmented extreme channel in a grid of subplots.
    Then, plot a single figure showing all five extreme segments overlaid
    on the overall channel boundary.

    Parameters:
       extreme_segments: dict returned by segment_channel_extreme.
       channel_polygon: shapely Polygon representing the overall channel boundary.
    """

    keys = list(extreme_segments.keys())
    n_extremes = len(keys)
    n_cols = int(np.ceil(np.sqrt(n_extremes)))
    n_rows = int(np.ceil(n_extremes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(keys):
        ax: Axes = axes[i]
        seg_poly, seg_mid, left_seg, right_seg, L_window = extreme_segments[key]
        seg_poly: Polygon
        x_seg, y_seg = seg_poly.exterior.xy
        start_line, end_line = get_end_closed_lines(seg_poly)

        # Plot the segmented channel
        ax.plot(x_seg, y_seg, '-r', lw=2, label=f"Segment: {key}")

        # Plot the corresponding midline
        ax.plot(seg_mid[:, 0], seg_mid[:, 1], '-b', lw=2, label="Midline")

        # Plot the end lines.
        ax.plot(start_line[:, 0], start_line[:, 1], '-', color='darkviolet', lw=2, label="Start End Line")
        ax.plot(end_line[:, 0], end_line[:, 1], '-g', lw=2, label="End End Line")

        ax.set_aspect('equal')
        ax.set_title(f"Extreme: {key}", fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(right=0.85)

    # Overlaid Plot: all segments on the overall channel
    fig2, ax2 = plt.subplots(figsize=(12, 10), constrained_layout=True)
    x_poly, y_poly = channel_polygon.exterior.xy
    ax2.plot(x_poly, y_poly, '--k', lw=1, label="Channel Boundary")
    ax2.fill(x_poly, y_poly, color='lightblue', alpha=0.3)

    colors = plt.cm.Set1(np.linspace(0, 1, n_extremes))
    for i, key in enumerate(keys):
        seg_poly, seg_mid, left_seg, right_seg, s_window = extreme_segments[key]
        x_seg, y_seg = seg_poly.exterior.xy
        ax2.plot(x_seg, y_seg, color=colors[i], lw=2, label=f"{key}")
        ax2.plot(seg_mid[:, 0], seg_mid[:, 1], 'b-', lw=2)

    ax2.set_aspect('equal')
    ax2.set_title("Overall Extreme Segments", fontsize=16)
    ax2.set_xlabel("x", fontsize=14)
    ax2.set_ylabel("y", fontsize=14)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

    fig2.subplots_adjust(right=0.85)
    plt.show()

def generate_fiber_in_segment(channel_seg_midline: npt.NDArray[np.float64],
                              L_fiber=5.0,
                              offset_factor=0.02,
                              N_fiber=50,
                              precise_offset=0.0):
    """
    Generate a fiber path inside a channel segment.

    Parameters:
        channel_seg_midline: ndarray of shape (M,2)
            Array of midline points for the segment. For straight channels, the shape is [2, 2], meaning start and end point
            xy position.
        L_fiber: float, optional
            Desired length of the fiber.
        offset_factor: float, optional
            Fraction of the total segment length at which the fiber begins.
            Do not use 0.0 because the wall pushes the robot away. Always leave some margin.
        N_fiber: int, optional
            Number of points along the fiber.
        precise_offset: float in absolute value.
            Precisely specify the distance of the starting tip of the robot to the starting line of the physical channel.
            This argument will supercede offset_factor if you provide non-0.0 input to this argument.

    Returns:
        fiber: ndarray of shape (3, N_fiber), where 3 is x, y, and z rows
            Fiber coordinates with a zero z-coordinate.
    """
    # Doesn't need backward modification because channel_seg_midline has the backward modification if backward
    # is True.

    # Compute cumulative arc-length along the segment midline.
    arc = np.cumsum(np.sqrt(np.sum(np.diff(channel_seg_midline, axis=0) ** 2, axis=1)))
    arc = np.insert(arc, 0, 0)
    total_length = arc[-1]

    # Determine the start and end positions along the arc.
    if precise_offset > 0.0:
        start = precise_offset
    else:
        start = offset_factor * total_length

    end = min(start + L_fiber, total_length)

    s_vals = np.linspace(start, end, N_fiber)

    # Create interpolation functions for x and y.
    interp_x = interp1d(arc, channel_seg_midline[:, 0])
    interp_y = interp1d(arc, channel_seg_midline[:, 1])

    fiber_x = interp_x(s_vals)
    fiber_y = interp_y(s_vals)

    # Return the fiber with z set to zero.
    # (xyz so shape 3, then N_fiber as columns). Row 0 is x. Row 1 is y
    fiber = np.vstack((fiber_x, fiber_y, np.zeros(N_fiber)))
    return fiber

def plot_channel_and_fiber(channel_polygon: Polygon,
                           seg_midline: npt.NDArray[np.float64],
                           fiber: npt.NDArray[np.float64],
                           title: str | None=None,
                           end_line_percent=1.0):
    """
    Plot the channel boundary, the segment midline, and the fiber within the segment.

    Parameters:
        channel_polygon: shapely Polygon
            The overall channel boundary.
        seg_midline: ndarray of shape (M,2)
            The midline points of the segment.
        fiber: ndarray of shape (3, N_fiber)
            The generated fiber coordinates.
        title: str, optional
            Title for the plot.
    """
    x_left_to_right_bound: npt.NDArray[np.float64] = np.array(channel_polygon.exterior.xy[0])
    y_left_to_right_bound: npt.NDArray[np.float64] = np.array(channel_polygon.exterior.xy[1])

    # Two points due to connecting both ends in width_construction()
    # Divide by half to get the end index for the left boundary
    x_left_bound = x_left_to_right_bound[:(x_left_to_right_bound.shape[0] - 2) // 2]
    y_left_bound = y_left_to_right_bound[:(x_left_to_right_bound.shape[0] - 2) // 2]

    x_right_bound = x_left_to_right_bound[-2:-x_left_bound.shape[0] - 2:-1]
    y_right_bound = y_left_to_right_bound[-2:-y_left_bound.shape[0] - 2:-1]

    index_end_line = int(x_right_bound.shape[0] * end_line_percent) - 1

    plt.figure(figsize=(8, 6))
    plt.plot(*channel_polygon.exterior.xy, 'k--', lw=1, label="Channel Boundary")
    plt.fill(*channel_polygon.exterior.xy, color='lightblue', alpha=0.3)
    plt.plot(
        [x_left_bound[index_end_line], x_right_bound[index_end_line]],
        [y_left_bound[index_end_line], y_right_bound[index_end_line]],
        'm-', lw=2, label="Specified End Line"
    )
    plt.plot(seg_midline[:, 0], seg_midline[:, 1], 'b-', lw=2, label="Midline")
    
    # Determine indices for the middle portion of the fiber
    N = fiber.shape[1]
    mid = N // 2

    # Plot fiber: beginning, middle, and end with different colors
    plt.plot(fiber[0, :], fiber[1, :], 'r-', lw=2, label="Fiber")
    plt.plot(fiber[0, mid], fiber[1, mid], 'go', lw=2, label="Fiber Middle")

    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_directors_from_positions(robot_init_pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute director frames for a straight rod given the nodal positions.
    The out-of-plane normal is set to [0, 0, 1] by default.

    Parameters
    ----------
    robot_init_pos : np.ndarray
        Array of shape (3, N_points) containing the positions of N nodes.

    Returns
    -------
    directors : np.ndarray
        Array of shape (3, 3, n_elements) where n_elements = N_points-1.
        For each element, directors[:,:,i] is an orthonormal 3x3 matrix with:
            - First row: the chosen normal vector,
            - Second row: the binormal (tangent x normal),
            - Third row: the tangent vector along the rod element.

        First row points outside of the screen.\n
        Second row points downward in the local frame of the robot.\n
        Third row points rightward (to the robot's end) in the local frame of the robot.\n

        On axis=1, the 3 denotes the normalized vector's component in xyz.
    """
    # Number of nodes and elements
    N = robot_init_pos.shape[1]
    n_elements = N - 1

    # Compute differences and tangents between consecutive nodes
    pos_diff = robot_init_pos[:, 1:] - robot_init_pos[:, :-1]
    rest_lengths = np.linalg.norm(pos_diff, axis=0)
    if np.any(rest_lengths < 1e-8):
        raise ValueError("Two consecutive nodes are too close together.")
    tangents = pos_diff / rest_lengths[np.newaxis, :]

    # Set default out-of-plane normal
    default_normal = np.array([0.0, 0.0, 1.0])

    # Initialize director array: shape (3, 3, n_elements)
    directors = np.zeros((3, 3, n_elements))

    for i in range(n_elements):
        t = tangents[:, i]

        # Check if the default normal is nearly parallel to the tangent.
        if abs(np.dot(t, default_normal)) > 0.99:
            # If too parallel, choose an alternate reference vector.
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = default_normal.copy()

        # Ensure the normal is normalized
        normal = normal / np.linalg.norm(normal)

        # First director: use the chosen normal
        # d1 out of screen
        d1 = normal

        # Second director: binormal = cross(tangent, normal)
        # d2 downward
        d2 = np.cross(t, d1)
        d2_norm = np.linalg.norm(d2)
        if d2_norm < 1e-8:
            raise ValueError(f"Computed binormal is too small at element {i}.")
        d2 /= d2_norm

        # Third director: tangent vector
        # d3 rightward
        d3 = t  # already normalized

        # Optionally, adjust d1 to ensure right-handedness: recompute d1 = cross(d2, d3)
        # d1 still out of screen
        d1 = np.cross(d2, d3)
        d1 /= np.linalg.norm(d1)

        # Store the director matrix for the i-th element.
        # Here we store the directors row-wise:
        #   Row 0: d1, Row 1: d2, Row 2: d3.
        directors[:, :, i] = np.vstack((d1, d2, d3))

        # Validate the orthonormality for this element.
        assert_allclose(np.linalg.norm(d1), 1.0, atol=1e-8,
                        err_msg=f"d1 is not unit length for element {i}.")
        assert_allclose(np.linalg.norm(d2), 1.0, atol=1e-8,
                        err_msg=f"d2 is not unit length for element {i}.")
        assert_allclose(np.linalg.norm(d3), 1.0, atol=1e-8,
                        err_msg=f"d3 is not unit length for element {i}.")
        # Check that d3 x d1 equals d2
        assert_allclose(np.cross(d3, d1), d2, atol=1e-8,
                        err_msg=f"Director frame for element {i} is not right-handed.")

    return directors