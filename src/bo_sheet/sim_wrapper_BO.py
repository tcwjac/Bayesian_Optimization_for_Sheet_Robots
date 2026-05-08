"""
Simulation wrapper functions for Bayesian optimization of magnetic beam robots.

This module provides wrapper functions for running magnetic beam simulations in channels,
supporting both friction coefficient optimization and arbitrary channel design optimization.
The simulations use the elastica framework with magnetic field interactions.

Author: Ziyu Ren
Date: 2025-07-07
"""

from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple
import itertools

import os
import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from elastica.timestepper import integrate_polygon_optimized
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from joblib import Parallel, delayed
from shapely.geometry import Polygon

from elastica import *
from magneto_pyelastica import *

from bo_sheet.interaction_channel import ChannelInteraction, _truncated_normal_sample_std1
import bo_sheet.channel_preprocess_utility as cu
from bo_sheet.curve_compute_utils import round_down

# Unit system: g-mm-ms-N-MPa
# Torque unit: N*mm = 1e-6 x [A/m] x [T] x [mm^3]
# Density unit: g/mm^3

# Simulation constants
DEFAULT_DAMPING_CONSTANT = 0.1
DEFAULT_BOUNDARY_STIFFNESS = 0.1
DEFAULT_CONTACT_DAMPING = 1e-10
DEFAULT_VELOCITY_THRESHOLD = 1e-2
DEFAULT_STEP_SKIP = 500

# Unit conversion factors
FORCE_TO_MICRO_NEWTON = 1e6  # N to μN
MS_TO_SECONDS = 1e-3  # ms to seconds

# Animation constants
ANIMATION_FPS = 15
ANIMATION_BITRATE = 1800
ANIMATION_INTERVAL = 100  # ms
PLOT_MARGIN = 5  # mm
MIN_DT_RATIO = 0.1  # Minimum dt as fraction of original
DT_REDUCTION_FACTOR = 0.2  # Reduce dt by 20% each iteration

# Material properties: (Young's modulus [MPa], Magnetization density [A/m], Density [kg/m^3])
MATERIAL_PROPERTIES = {
    'PDMS_1_1': (0.900726810, 70314.28, 0.001638991),
    'PDMS_1_2': (1.597738235, 114333.64, 0.002363737),
    'PDMS_1_3': (3.823944749, 152678.40, 0.00249899),
    'PDMS_1_4': (4.791731288, 217961.05, 0.003183369),
    'ecoflex0010_1_1': (0.135666, 83301.98, 0.00200615),
    'ecoflex0010_1_2': (0.210838684, 108479.33, 0.002191092),
    'ecoflex0010_1_3': (0.339161436, 168107.87, 0.002847701)
}

# Friction coefficients: (static_wall, kinetic_wall, static_wall_end, kinetic_wall_end, static_substrate, kinetic_substrate)
FRICTION_COEFFICIENTS = {
    'PDMS_1_2': (
        14.057105926375833, 8.174926351320835, 10.106913860902413,
        3.5913695226435034, 0.20625551949932347, 2.671900314627418e-13
    ),
    'ecoflex0010_1_2': (
        14.057105926375833, 8.174926351320835, 10.106913860902413,
        3.5913695226435034, 0.20625551949932347, 2.671900314627418e-13
    ),
}

class MagneticBeamSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    """
    Inheriting elastica's base simulator classes with Constraints, Forcing, Damping, and CallBacks
    functionalities.
    """
    pass

class MagneticBeamCallBack(CallBackBaseClass):
    """
    Callback class for collecting simulation data during magnetic beam simulation.
    
    This class implements the elastica callback interface to collect various
    simulation metrics at specified intervals, including energies, positions,
    forces, and velocities.
    
    Attributes:
        every (int): Number of simulation steps between data collection calls
        callback_params (Dict[str, List]): Dictionary storing collected data arrays
    """
    
    def __init__(self, step_skip: int, callback_params: Dict[str, List]) -> None:
        """
        Initialize the callback with collection parameters.
        
        Args:
            step_skip: Number of steps to skip between data collection
            callback_params: Dictionary to store collected data
        """
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system: CosseratRod, time: float, current_step: int) -> None:
        """
        Collect simulation data at specified intervals.
        
        This method is called by the elastica framework during simulation.
        It collects various energy components, position data, forces, and
        velocities when the current step is a multiple of the skip interval.
        
        Args:
            system: The elastica rod system object
            time: Current simulation time [ms]
            current_step: Current simulation step number
        """
        if current_step % self.every == 0:
            # Collect energy components
            self.callback_params["te"].append(system.compute_translational_energy())
            self.callback_params["re"].append(system.compute_rotational_energy())
            self.callback_params["se"].append(system.compute_shear_energy())
            self.callback_params["be"].append(system.compute_bending_energy())
            
            # Collect time and position data
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            
            # Collect force data (convert to micro-Newtons for better scaling)
            self.callback_params["external_force"].append(
                system.external_forces * FORCE_TO_MICRO_NEWTON
            )
            
            # Collect additional simulation metrics
            self.callback_params["step"].append(current_step)
            self.callback_params["velocity_norm"].append(
                np.linalg.norm(system.velocity_collection)
            )

def _validate_simulation_inputs(
    dt: float,
    final_time: float,
    length: float,
    beam_width: float,
    beam_thickness: float,
    B_field: float,
    B_frequency: float,
    variation_level: float
) -> None:
    """Validate simulation input parameters."""
    if dt <= 0 or final_time <= 0:
        raise ValueError("Time parameters (dt, final_time) must be positive.")
    if length <= 0 or beam_width <= 0 or beam_thickness <= 0:
        raise ValueError("Beam dimensions must be positive.")
    if B_field <= 0 or B_frequency <= 0:
        raise ValueError("Magnetic field parameters must be positive.")
    if variation_level <= 0 and variation_level > 1:
        raise ValueError("Friction coefficient variation must be 0 <= variation_level < 1.")

def run_crawling_sim(
    length: float,
    beam_width: float,
    beam_thickness: float,
    n_waveform: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_polygon: Polygon,
    dist_threshold: float,
    robot_init_pos: npt.NDArray[np.float64],
    rod_origin: npt.NDArray[np.float64],
    rod_director: npt.NDArray[np.float64],
    B_field: float,
    B_frequency: float,
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    magnetization_direction=np.array([0.0], np.float64),
    variation_level=0.0,
    dynamic_fric_coeff=False,
    dynamic_fric_variation_level=0.0,
    backward=False,
    move_end_line_to=1.0,
    bifurcation=False,
    right_endline=0.0,
    left_endline=0.0,
    n_elem=0
) -> Tuple[Dict[str, List], np.float64, str]:
    """
    Run a single magnetic beam crawling simulation in a channel.
    
    This function simulates a magnetic beam robot moving through a channel under
    the influence of a rotating magnetic field. The beam experiences friction
    with channel walls and substrate, and may have variations in properties.
    
    Units: g-mm-ms-N-MPa system
    - Length: mm
    - Time: ms
    - Force: N
    - Pressure: MPa
    - Torque: N·mm = 1e-6 × [A/m] × [T] × [mm³]
    
    Args:
        length: Beam length [mm]
        beam_width: Beam width [mm]
        beam_thickness: Beam thickness [mm]
        n_waveform: Number of magnetization waveform periods
        beam_density: Beam material density [g/mm³]
        modulus: Young's modulus [MPa]
        magnetization_density: Magnetization density [A/m]
        channel_polygon: Channel geometry as Shapely Polygon
        dist_threshold: Distance threshold for contact interactions [mm]
        robot_init_pos: Initial beam positions [3 × (n_elem+1)]
        rod_origin: Rod origin point [3]
        rod_director: Rod director matrix [3 × 3 × n_elem]
        B_field: Magnetic field strength [T]
        B_frequency: Magnetic field frequency [Hz]
        kinetic_mu_substrate_array: Kinetic friction coefficients for substrate. Shape (n_elem + 1,)
        static_mu_substrate_array: Static friction coefficients for substrate. Shape (n_elem + 1,)
        kinetic_mu_wall_array: Kinetic friction coefficients for walls. Shape (n_elem + 1,)
        static_mu_wall_array: Static friction coefficients for walls. Shape (n_elem + 1,)
        dt: Simulation time step [ms]
        final_time: Final simulation time [ms]
        variation_level: Friction coefficient variation level [0,1)
        backward: Crawl backward
        
    Returns:
        Tuple:
            - Dictionary with simulation data (energies, positions, forces, etc.)
            - Actual final simulation time [ms]
            
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If simulation fails to converge
    """

    _validate_simulation_inputs(
        dt, final_time, length, beam_width, beam_thickness, B_field, B_frequency, variation_level
    )
    
    original_dt = dt
    min_dt = MIN_DT_RATIO * original_dt
    simulation_successful = False
    last_exception = None
    
    # Attempt simulation with adaptive time stepping
    while dt >= min_dt and not simulation_successful:
        try:
            # Calculate equivalent cylindrical rod parameters (beam to rod)
            r_equivalent = (beam_width * beam_thickness ** 3 / (3 * np.pi)) ** 0.25
            density_equivalent = beam_width * beam_thickness * beam_density / (np.pi * r_equivalent ** 2)

            poisson_ratio = 0.5
            shear_modulus = modulus / (2 * (1 + poisson_ratio))
            
            # Set rotating magnetic field
            angular_frequency = -2 * np.pi * B_frequency * 1e-3
            magnetic_field = B_field * 1e-6

            magnetic_beam_sim = MagneticBeamSimulator()
            
            magnetic_rod = CosseratRod.straight_rod(
                n_elem,
                start=rod_origin.reshape(3),
                direction=np.array([1.0, 0.0, 0.0]),
                directors=rod_director,
                normal=np.array([0.0, 0.0, 1.0]),
                base_length=length,
                base_radius=r_equivalent,
                density=density_equivalent,
                youngs_modulus=modulus,
                shear_modulus=shear_modulus,
                position=robot_init_pos,
            )
            magnetic_beam_sim.append(magnetic_rod)
            
            magnetic_beam_sim.constrain(magnetic_rod).using(FreeBC)
            
            magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
                ChannelInteraction,
                boundary_stiffness=DEFAULT_BOUNDARY_STIFFNESS,
                contact_damping=DEFAULT_CONTACT_DAMPING,
                dist_threshold=dist_threshold,
                velocity_threshold=DEFAULT_VELOCITY_THRESHOLD,
                kinetic_mu_substrate_array=kinetic_mu_substrate_array,
                static_mu_substrate_array=static_mu_substrate_array,
                kinetic_mu_wall_array=kinetic_mu_wall_array,
                static_mu_wall_array=static_mu_wall_array,
                channel_polygon=channel_polygon,
                total_gravitational_force=beam_width * beam_thickness * length * beam_density * 0.00981,
                variation_level=variation_level,
                dynamic_fric_coeff=dynamic_fric_coeff,
                dynamic_fric_variation_level=dynamic_fric_variation_level
            )
            
            if backward != True:
                magnetic_field_object = SingleModeOscillatingMagneticField(
                    magnetic_field_amplitude=magnetic_field * np.array([1, 1, 0]),
                    magnetic_field_angular_frequency=np.array(
                        [angular_frequency, angular_frequency, 0.0]
                    ),
                    magnetic_field_phase_difference=np.array([0.0, np.pi / 2, 0.0]),
                    ramp_interval=100.0 * dt,
                    start_time=0.0,
                    end_time=100.0 * final_time,
                )
            else:
                magnetic_field_object = SingleModeOscillatingMagneticField(
                    magnetic_field_amplitude=magnetic_field * np.array([-1, -1, 0]),
                    magnetic_field_angular_frequency=np.array(
                        [angular_frequency, angular_frequency, 0.0]
                    ),
                    magnetic_field_phase_difference=np.array([0.0, np.pi / 2, 0.0]),
                    ramp_interval=100.0 * dt,
                    start_time=0.0,
                    end_time=100.0 * final_time,
                )

            magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
                MagneticForces,
                external_magnetic_field=magnetic_field_object,
                magnetization_density=magnetization_density,
                magnetization_direction=magnetization_direction,
                rod_volume=magnetic_rod.volume,
                rod_director_collection=magnetic_rod.director_collection,
            )
            
            magnetic_beam_sim.dampen(magnetic_rod).using(
                AnalyticalLinearDamper,
                damping_constant=DEFAULT_DAMPING_CONSTANT,
                time_step=dt,
            )
            
            post_processing_dict = defaultdict(list)
            magnetic_beam_sim.collect_diagnostics(magnetic_rod).using(
                MagneticBeamCallBack,
                step_skip=DEFAULT_STEP_SKIP,
                callback_params=post_processing_dict
            )
            
            magnetic_beam_sim.finalize()
            timestepper = PositionVerlet()
            total_steps = int(final_time / dt)
            
            # reached_which_endline can be 'neither' which means too slow or curled into itself
            actual_final_time_in_ms, reached_which_endline = integrate_polygon_optimized(
                timestepper, magnetic_beam_sim, final_time, channel_polygon,
                total_steps, progress_bar=True, backward=backward, move_end_line_to=move_end_line_to,
                bifur=bifurcation, right_endline=right_endline, left_endline=left_endline
            )

            actual_final_time_in_ms: np.float64
            reached_which_endline: str

            simulation_successful = True
            
        except Exception as e:
            last_exception = e
            dt -= original_dt * DT_REDUCTION_FACTOR
    
    # Check if simulation succeeded
    if not simulation_successful:
        error_msg = (
            f"Simulation failed to converge after reducing dt to {dt:.4f} "
            f"({MIN_DT_RATIO*100}% of original {original_dt:.4f}). "
            f"Last error: {str(last_exception)}"
        )
        raise RuntimeError(error_msg)
    
    return post_processing_dict, actual_final_time_in_ms, reached_which_endline

def _travel_distance_along_midline(channel_x_mid: npt.NDArray[np.float64],
                                   channel_y_mid: npt.NDArray[np.float64],
                                   positions_over_time: npt.NDArray[np.float64],
                                   move_end_line_to=1.0) -> np.float64:
    """
    Calculate the travel distance along the channel midline using spline projection.
    The computed distance is independent of the robot length because the middle node
    of the robot start at the same position and we only consider the middle node travel
    distance.
    
    1. Creates a parametric spline representation of the channel midline (for curved channels)
       OR uses simple line projection (for straight channels with 2 points)
    2. Projects robot center node's position onto the closest point on the path
    3. Calculates cumulative arc length along the path
    4. Measures net progress along the intended path

    Parameters
    ----------
    channel_x_mid : 1D np.ndarray
        x-coordinates of channel midline (the intended path)
    channel_y_mid : 1D np.ndarray
        y-coordinates of channel midline (the intended path)
    positions_over_time : np.ndarray
        Array of robot positions over time [n_frames, xyz, n_points]
        Track the first point (robot head) position

    Returns
    -------
    np.float64
        Net distance traveled along the midline [mm]
        Positive denotes forward progress along the intended path
    """
    # Check the number of midline points to determine if it's a straight channel
    num_midline_points = len(channel_x_mid)
    
    # Track the robot center point over time
    n_robot_points = positions_over_time.shape[2]
    center_index = n_robot_points // 2
    # Shape [n_frames, xy] of the middle node of the robot
    point_positions: npt.NDArray[np.float64] = positions_over_time[:, :2, center_index]
    
    if num_midline_points == 2:
        # STRAIGHT CHANNEL CASE: Use simple line projection
        # The midline is just a straight line between two points
        
        # Define the straight line from start to end
        start_point: npt.NDArray[np.float64] = np.array([channel_x_mid[0], channel_y_mid[0]])
        end_point: npt.NDArray[np.float64] = np.array([channel_x_mid[1], channel_y_mid[1]])

        # Line vector and total length
        line_vector: npt.NDArray[np.float64] = (move_end_line_to * end_point) - start_point
        total_length: np.float64 = np.linalg.norm(line_vector)
        line_unit_vector: npt.NDArray[np.float64] = line_vector / total_length if total_length > 0 else np.array([1, 0])
        
        # Project robot's center node position over time onto the line
        projected_distances = np.empty(len(point_positions))
        for i, point in enumerate(point_positions):
            point: npt.NDArray[np.float64] # Shape (2,)
            # Vector from line start to point
            point_vector: npt.NDArray[np.float64] = point - start_point
            # Project onto line direction (dot product gives signed distance along line)
            projected_distances[i] = np.dot(point_vector, line_unit_vector)
        
        # Calculate net travel distance
        net_distance: np.float64 = projected_distances[-1] - projected_distances[0]
        
    else:
        # CURVED CHANNEL CASE: Use spline interpolation
        # Need at least 4 points for cubic spline
        if num_midline_points < 4:
            raise ValueError(f"Curved channels need at least 4 midline points, got {num_midline_points}")
        
        if move_end_line_to < 1.0:
            index_length: int = channel_x_mid.shape[0]

            target_end_index = int(move_end_line_to * index_length)

            channel_x_mid = channel_x_mid[:target_end_index]
            channel_y_mid = channel_y_mid[:target_end_index]

        # Create parametric spline representation of the channel midline
        # splprep creates a B-spline representation with parameter u in [0,1]
        tck, _ = splprep([channel_x_mid, channel_y_mid], s=0)

        # Generate dense sampling for accurate distance calculations
        # Create 1000 points along the spline to compute cumulative arc length
        dense_u: npt.NDArray[np.float64] = np.linspace(0, 1, 1000)
        dense_coords = np.array(splev(dense_u, tck)).T  # [1000 x 2] array of (x,y) coords
        
        # Calculate cumulative arc length along the spline
        segment_lengths: npt.NDArray[np.float64] = np.sqrt(np.sum(np.diff(dense_coords, axis=0) ** 2, axis=1))
        dense_cum: npt.NDArray[np.float64] = np.concatenate(([0], np.cumsum(segment_lengths)))

        # Define distance function for point-to-spline projection
        @np.vectorize
        def dist_sq(u_val, point_x, point_y):
            """Calculate squared distance from point to spline at parameter u_val"""
            x, y = splev(u_val, tck)  # Get spline coordinates at parameter u_val
            return (x - point_x) ** 2 + (y - point_y) ** 2

        # Project robot center node's position over time onto the spline
        # For each time step, find the parameter u that minimizes distance to spline
        projected_u = np.empty(len(point_positions))
        for i, point in enumerate(point_positions):
            # This finds the closest point on the spline to the robot position
            res = minimize(
                lambda u: dist_sq(u, point[0], point[1]),
                x0=0.5,  # Start search from middle of spline
                bounds=[(0, 1)],  # Constrain to valid parameter range
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            projected_u[i] = res.x[0]

        # Convert parameter values to cumulative distances
        # Map u-parameters to actual arc length distances along the spline
        projected_cum = np.interp(projected_u, dense_u, dense_cum)

        # Calculate net travel distance
        # Difference in cumulative distance = net progress along the path
        net_distance = projected_cum[-1] - projected_cum[0]

    return net_distance

def _check_self_crossings(positions_over_time: npt.NDArray[np.float64]) -> List[int]:
    """
    Checks for self-crossings in the 2D (XY) projection of a curve over multiple frames.

    This function iterates through each time frame in the input array,
    projects the curve onto the XY plane, and checks for self-intersections.
    A self-intersection occurs if any two non-adjacent segments of the curve cross each other.

    Parameters
    ----------
    positions_over_time : numpy.ndarray
        A 3D numpy array of shape (num_frames, 3, num_points), representing the
        coordinates of the curve's points over time. The z-coordinate is ignored.

    Returns
    -------
    list of int
        A list of frame indices where self-crossings are detected.
    """
    crossing_frames = []
    num_frames = positions_over_time.shape[0]

    for i in range(num_frames):
        # Extract the 2D projection (x, y coordinates)
        curve_points: npt.NDArray[np.float64] = positions_over_time[i, :2, :].T  # Shape: (num_points, 2)

        n_points = curve_points.shape[0]
        if n_points < 4:
            continue

        # Create segments from points
        segments = np.array([curve_points[:-1], curve_points[1:]]).transpose(1, 0, 2)
        n_segments = len(segments)

        # Get indices for pairs of non-adjacent segments.
        # k=2 avoids checking adjacent segments, which always "intersect" at their shared point.
        idx1, idx2 = np.triu_indices(n_segments, k=2)

        # If there are no non-adjacent pairs, continue
        if len(idx1) == 0:
            continue

        # Get segment endpoints for all pairs
        p1, p2 = segments[idx1, 0], segments[idx1, 1]
        p3, p4 = segments[idx2, 0], segments[idx2, 1]

        # Calculate vectors for line segments
        v1 = p2 - p1
        v2 = p4 - p3

        # Using 2D cross product to check orientation
        # (p - q) x r = (px - qx) * ry - (py - qy) * rx
        cross_product = lambda p, q, r: (p[:, 0] - q[:, 0]) * r[:, 1] - (p[:, 1] - q[:, 1]) * r[:, 0]

        d1 = cross_product(p3, p1, v1)
        d2 = cross_product(p4, p1, v1)
        d3 = cross_product(p1, p3, v2)
        d4 = cross_product(p2, p3, v2)

        # Check for intersection.
        # This occurs if orientations are different (d1*d2 < 0 and d3*d4 < 0).
        # np.finfo(float).eps is used to handle floating point inaccuracies near zero.
        intersect = (d1 * d2 < -np.finfo(float).eps) & (d3 * d4 < -np.finfo(float).eps)

        if np.any(intersect):
            crossing_frames.append(i)

    return crossing_frames

def _calculate_robot_speed_center_point(
    positions_over_time: npt.NDArray[np.float64],
    actual_final_time_in_ms: np.float64,
    channel_x_mid: npt.NDArray[np.float64],
    channel_y_mid: npt.NDArray[np.float64]
) -> np.float64:
    """
    Calculate robot speed based on center point travel distance along the channel midline.
    
    This function works for both straight and curved channels by utilizing the 
    _travel_distance_along_midline function. It omits the transition period
    (first 1 second) to measure steady-state locomotion speed.
    
    The function automatically determines whether the channel is straight or curved
    based on the number of midline points:
    - 2 points: Straight channel (uses line projection)
    - 4+ points: Curved channel (uses spline interpolation)
    
    Args:
        positions_over_time: Array of robot positions over time [frames, 3, n_nodes]
        actual_final_time: Time it took for the robot to reach the end [ms]
        channel_x_mid: x-coordinates of channel midline (REQUIRED for accurate calculation)
        channel_y_mid: y-coordinates of channel midline (REQUIRED for accurate calculation)
        
    Returns:
        np.float64: Robot speed [mm/s] calculated from travel distance along midline
    """

    total_time = actual_final_time_in_ms * MS_TO_SECONDS
    total_frames = positions_over_time.shape[0]
    fps = total_frames / total_time
    # fps is implicitly multiplied by 1 for if checking.
    if total_frames <= fps:
        return np.float64(0.0)
    
    # Shape (n_frames - (fps+1) or (fps+2), 3, n_nodes)
    # n_frames are always over 1000, so at worst taking 2 frames away is negligible in the indexing
    stable_period_positions: npt.NDArray[np.float64] = positions_over_time[int(fps) + 1:]

    # Not stable at all
    if stable_period_positions.shape[0] == 0:
        return np.float64(0.0)

    distance_traveled = _travel_distance_along_midline(
        channel_x_mid, channel_y_mid, stable_period_positions
    )
    total_stable_time = stable_period_positions.shape[0] / fps
    speed = abs(distance_traveled) / total_stable_time if total_stable_time > np.float64(0.0) else np.float64(0.0)
    return speed

def _create_simulation_movie(
    positions_over_time: npt.NDArray[np.float64],
    channel_polygon: Polygon,
    movie_name: str,
    save_directory: str
) -> None:
    """
    Create an animated movie visualization of the simulation.
    
    Args:
        positions_over_time: Array of robot positions over time [frames, 3, n_points]
        channel_polygon: Channel geometry for background
        movie_name: Output movie filename
    """
    print(f"Creating {movie_name} movie")

    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.dpi
    width, height = int(width), int(height)

    video_path = os.path.join(save_directory, movie_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, ANIMATION_FPS, (width, height))

    x_poly, y_poly = channel_polygon.exterior.xy
    n_frames = positions_over_time.shape[0]

    for frame in range(n_frames):
        ax.clear()
        ax.set_aspect('equal')
        ax.plot(x_poly, y_poly, 'k-', linewidth=2, label="Channel")

        current_pos = positions_over_time[frame, :2, :]
        ax.plot(current_pos[0, :], current_pos[1, :], 'r-', linewidth=2, label="Robot")
        ax.set_xlim(min(x_poly) - PLOT_MARGIN, max(x_poly) + PLOT_MARGIN)
        ax.set_ylim(min(y_poly) - PLOT_MARGIN, max(y_poly) + PLOT_MARGIN)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_title(movie_name)

        canvas.draw()
        
        img = np.asarray(canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)
    print(f"Video saved to {video_path}")

def render_clean_plate_png(
    channel_polygon: Polygon,
    movie_name: str,
    save_directory: str,
    figsize=(10, 6),
    dpi=None,
    include_title=True
):
    """
    Render a frame-independent clean plate PNG (channel only, no legend, no robot).
    """
    os.makedirs(save_directory, exist_ok=True)
    png_path = os.path.join(save_directory, f"{movie_name}_clean_plate.png")

    fig, ax = plt.subplots(figsize=figsize)
    if dpi is not None:
        fig.set_dpi(dpi)

    canvas = FigureCanvas(fig)

    x_poly, y_poly = channel_polygon.exterior.xy
    ax.set_aspect('equal')
    ax.plot(x_poly, y_poly, 'k-', linewidth=2)

    ax.set_xlim(min(x_poly) - PLOT_MARGIN, max(x_poly) + PLOT_MARGIN)
    ax.set_ylim(min(y_poly) - PLOT_MARGIN, max(y_poly) + PLOT_MARGIN)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')

    if include_title:
        ax.set_title(movie_name)

    canvas.draw()

    fig.savefig(png_path, dpi=fig.dpi)
    plt.close(fig)

def _wrapper_for_evaluation_function(
    length: float,
    beam_width: float,
    beam_thickness: float,
    n_waveform: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_polygon: Polygon,
    channel_x_mid: npt.NDArray[np.float64],
    channel_y_mid: npt.NDArray[np.float64],
    dist_threshold: float,
    robot_init_pos: npt.NDArray[np.float64],
    robot_init_origin: npt.NDArray[np.float64],
    robot_init_director: npt.NDArray[np.float64],
    B_field: float,
    B_frequency: float,
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    magnetization_direction: npt.NDArray[np.float64],
    n_elem: int,
    variation_level=0.0,
    dynamic_fric_coeff=False,
    backward=False
) -> np.float64:

    post_processing_dict, actual_final_time_in_ms, _ = run_crawling_sim(
        length=length,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        n_waveform=n_waveform,
        beam_density=beam_density,
        modulus=modulus,
        magnetization_density=magnetization_density,
        channel_polygon=channel_polygon,
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos,
        rod_origin=robot_init_origin,
        rod_director=robot_init_director,
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        magnetization_direction=magnetization_direction,
        variation_level=variation_level,
        dynamic_fric_coeff=dynamic_fric_coeff,
        backward=backward,
        n_elem=n_elem
    )

    # Get positions
    positions_over_time = np.array(post_processing_dict["position"])

    # Catch unexpected compute error from sim here
    if np.isnan(positions_over_time).any() or positions_over_time.shape[0] == 0 or actual_final_time_in_ms == np.float64(0.0):
        return np.float64(0.0)

    # Compute speed
    speed = _calculate_robot_speed_center_point(positions_over_time, actual_final_time_in_ms, channel_x_mid, channel_y_mid)
    # Check self crossing
    crossing_frames = _check_self_crossings(positions_over_time)
    cross_ratio = len(crossing_frames) / positions_over_time.shape[0]

    if cross_ratio > 0.0:
        speed = np.clip(speed * (1 - 400 * (cross_ratio ** 2)), a_min=0.0, a_max=None)

    return speed

def friction_objective_function(
    BO_parameters: Dict[str, float],
    fixed_parameters: Dict[str, Polygon | List[float] | int | float | npt.NDArray[np.float64] | bool]
) -> Dict[str, float]:
    """
    Evaluate simulation performance for friction coefficient optimization.
    
    Args:
        BO_parameters: Dictionary containing optimization parameters:
            - static_mu_wall: Static friction coefficient for channel walls
            - kinetic_mu_wall: Kinetic friction coefficient for channel walls
            - static_mu_wall_end: Static friction coefficient for beam ends on walls
            - kinetic_mu_wall_end: Kinetic friction coefficient for beam ends on walls
            - static_mu_substrate: Static friction coefficient for substrate
            - kinetic_mu_substrate: Kinetic friction coefficient for substrate
            - k_modulus: Young's modulus scaling factor
            - k_mag_den: Magnetization density scaling factor
        fixed_parameters: Dictionary containing fixed simulation parameters:
            - channel_list: List of channel polygons
            - channel_x_mid_list: List of polygons' midlines projected onto Cartestian x
            - channel_y_mid_list: List of polygons' midlines projected onto Cartestian y
            - exp_speed_list: List of experimental speeds for comparison
            - robot_length: Beam length [mm]
            - robot_width: Beam width [mm]
            - robot_thickness: Beam thickness [mm]
            - robot_n_waveform: Number of magnetization waveform periods
            - robot_density: Beam material density [g/mm³]
            - robot_modulus: Young's modulus [MPa]
            - robot_magnetization_density: Magnetization density [A/m]
            - dist_threshold: Distance threshold for contact interactions [mm]
            - robot_n_elem: Number of segments between every consecutive node pair of the robot
            - initial_position: Initial position of the robot. Shape (xyz, n_elem+1), so (3, n_elem+1)
            - initial_origin: Initial position of the 0th node of the robot. Shape (xyz,)
            - initial_director: Initial frame definition of the robot's nodes. Shape (3, 3, n_node).\\
              Axis 0 denotes out-of-screen, downward, and rightward vectors which form the robot's nodes local frames.\\
              Axis 1 denotes the xyz components as orthonormal vectors.
            - B_field: Magnetic field strength [T]
            - B_frequency: Magnetic field frequency [Hz]
            - dt: Simulation time step [ms]
            - final_time: Final simulation time [ms]
            - whether_output_movie: Boolean for movie generation
            - variation_level: Friction coefficient variation level [0, 1)
            - backward: Crawl backward
            
    Returns:
        float: Total weighted difference between experimental and simulated speeds
        
    Raises:
        ValueError: If required parameters are missing or dimensions don't match
        RuntimeError: If all simulations fail
    """
    required_bo_params = {
        'static_mu_wall', 'kinetic_mu_wall', 'static_mu_wall_end',
        'kinetic_mu_wall_end', 'static_mu_substrate', 'kinetic_mu_substrate' ,'k_mag_den', 'k_modulus'
    }
    
    missing_params = required_bo_params - set(BO_parameters.keys())
    if missing_params:
        raise ValueError(f"Missing required BO parameters: {missing_params}")

    static_mu_wall: float = BO_parameters['static_mu_wall']
    kinetic_mu_wall: float = BO_parameters['kinetic_mu_wall']
    static_mu_wall_end: float = BO_parameters['static_mu_wall_end']
    kinetic_mu_wall_end: float = BO_parameters['kinetic_mu_wall_end']
    static_mu_substrate: float = BO_parameters['static_mu_substrate']
    kinetic_mu_substrate: float = BO_parameters['kinetic_mu_substrate']
    k_mag_den: float = BO_parameters['k_mag_den']
    k_modulus: float = BO_parameters['k_modulus']

    channel_polygon_list: List[Polygon] = fixed_parameters['channel_list']
    channel_x_mid_list: List[npt.NDArray[np.float64]] = fixed_parameters['channel_x_mid_list']
    channel_y_mid_list: List[npt.NDArray[np.float64]] = fixed_parameters['channel_y_mid_list']
    exp_speed_list: List[float] = fixed_parameters['exp_speed_list']
    exp_std_list: List[float] = fixed_parameters['exp_std_list']
    robot_length: float = fixed_parameters['robot_length']
    robot_width: float = fixed_parameters['robot_width']
    robot_thickness: float = fixed_parameters['robot_thickness']
    robot_n_waveform: float = fixed_parameters['robot_n_waveform']
    robot_density: float = fixed_parameters['robot_density']
    robot_modulus: float = fixed_parameters['robot_modulus'] * k_modulus
    robot_magnetization_density: float = fixed_parameters['robot_magnetization_density'] * k_mag_den
    dist_threshold: float = fixed_parameters['dist_threshold']
    robot_n_elem: int = fixed_parameters['robot_n_elem']
    robot_init_pos_list: List[npt.NDArray[np.float64]] = fixed_parameters['robot_init_pos_list']
    robot_init_origin_list: List[npt.NDArray[np.float64]] = fixed_parameters['robot_init_origin_list']
    robot_init_director_list: List[npt.NDArray[np.float64]] = fixed_parameters['robot_init_director_list']
    B_field: float = fixed_parameters['B_field']
    B_frequency: float = fixed_parameters['B_frequency']
    dl: float = fixed_parameters['dl']
    cfl: float = fixed_parameters['cfl']
    final_time: float = fixed_parameters['final_time']
    save_directory: str = fixed_parameters['save_directory']
    variation_level: float = fixed_parameters.get('variation_level', 0.0)
    backward: bool = fixed_parameters.get('backward', False)
    mag_dir: npt.NDArray[np.float64] = fixed_parameters['mag_dir']

    if len(channel_polygon_list) != len(exp_speed_list):
        raise ValueError(
            f"Channel count ({len(channel_polygon_list)}) must match "
            f"experimental speed count ({len(exp_speed_list)})"
        )
    
    if len(channel_polygon_list) != len(channel_x_mid_list) or len(channel_polygon_list) != len(channel_y_mid_list):
        raise ValueError(
            f"Channel polygon count ({len(channel_polygon_list)}) must match "
            f"channel_x_mid count ({len(channel_x_mid_list)}) and "
            f"channel_y_mid count ({len(channel_y_mid_list)})"
        )
    
    if len(exp_std_list) != len(exp_speed_list):
        raise ValueError(f"exp_std_list length ({len(exp_std_list)}) must match exp_speed_list length ({len(exp_speed_list)})")
    
    dt = round_down(np.float64(cfl * dl / np.sqrt(robot_modulus / robot_density)))
    
    kinetic_mu_substrate_array = np.full((robot_n_elem + 1,), kinetic_mu_substrate)
    static_mu_substrate_array = np.full((robot_n_elem + 1,), static_mu_substrate)
    kinetic_mu_wall_array = np.full((robot_n_elem + 1,), kinetic_mu_wall)
    static_mu_wall_array = np.full((robot_n_elem + 1,), static_mu_wall)

    kinetic_mu_wall_array[:1] = kinetic_mu_wall_end
    kinetic_mu_wall_array[-1:] = kinetic_mu_wall_end
    static_mu_wall_array[:1] = static_mu_wall_end
    static_mu_wall_array[-1:] = static_mu_wall_end

    # Use n_jobs=1 for single simulation
    # Use n_jobs=-1 for multiple concurrent simulations
    n_jobs = 1 if len(channel_polygon_list) == 1 else -1
    
    joblib_results = Parallel(n_jobs=n_jobs)(delayed(_wrapper_for_evaluation_function)(
        length=robot_length,
        beam_width=robot_width,
        beam_thickness=robot_thickness,
        n_waveform=robot_n_waveform,
        beam_density=robot_density,
        modulus=robot_modulus,
        magnetization_density=robot_magnetization_density,
        channel_polygon=channel_polygon_list[idx],
        channel_x_mid=channel_x_mid_list[idx],
        channel_y_mid=channel_y_mid_list[idx],
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos_list[idx],
        robot_init_origin=robot_init_origin_list[idx],
        robot_init_director=robot_init_director_list[idx],
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        magnetization_direction=mag_dir,
        n_elem=robot_n_elem,
        variation_level=variation_level,
        backward=backward
        ) for idx in range(len(channel_polygon_list))
    )

    sim_speeds: List[np.float64] = joblib_results
    
    worst_score = np.float64(1.2e2)

    if any(speed == np.float64(0.0) for speed in sim_speeds):
        total_difference = worst_score
        objectives = {
            "totalDifference": total_difference,
            'con': total_difference
        }
        with open(os.path.join(save_directory, 'debug.txt'), 'a') as file:
            file.write(f"In sim_wrapper, early termination: {objectives}\n")
            file.close()
        return objectives

    weighted_level_difference = 0.0
    weighted_trend_difference = 0.0

    for exp, sim in zip(exp_speed_list, sim_speeds):
        weighted_diff = np.abs(exp - sim)

        weighted_level_difference += weighted_diff
        
        with open(os.path.join(save_directory, 'debug.txt'), 'a') as file:
            file.write(f"In sim_wrapper, sim: {sim:.3f}. exp: {exp:.3f}. weighted_diff: {weighted_diff:.3f}\n")
            file.close()

    for i in range(1, len(exp_speed_list)):
        exp_trend = exp_speed_list[i] - exp_speed_list[i - 1]
        sim_trend = sim_speeds[i] - sim_speeds[i - 1]

        weighted_trend_diff = np.abs(exp_trend - sim_trend)

        weighted_trend_difference += weighted_trend_diff

        with open(os.path.join(save_directory, 'debug.txt'), 'a') as file:
            file.write(f"In sim_wrapper, sim_trend: {sim_trend:.3f}. exp_trend: {exp_trend:.3f}. weighted_trend_diff: {weighted_trend_diff:.3f}\n")
            file.close()

    abs_weight = 1.0
    trend_weight = 3.0

    total_difference = abs_weight * weighted_level_difference + trend_weight * weighted_trend_difference

    if total_difference < np.float64(0.0) or np.isnan(total_difference) or total_difference > worst_score:
        total_difference = worst_score

    objectives = {
        "totalDifference": total_difference,
        'con': total_difference
    }

    with open(os.path.join(save_directory, 'debug.txt'), 'a') as file:
        file.write(f"In sim_wrapper, objectives: {objectives}\n")
        file.write(f"static_mu_wall: {static_mu_wall}\n")
        file.write(f"kinetic_mu_wall: {kinetic_mu_wall}\n") 
        file.write(f"static_mu_wall_end: {static_mu_wall_end}\n")
        file.write(f"kinetic_mu_wall_end: {kinetic_mu_wall_end}\n")
        file.write(f"static_mu_substrate: {static_mu_substrate}\n")
        file.write(f"kinetic_mu_substrate: {kinetic_mu_substrate}\n")
        file.write(f"k_modulus: {k_modulus}\n")
        file.write(f"k_mag_den: {k_mag_den}\n")
        file.close()

    return objectives

def _fric_var_1std_for_sim_wrapper(
    robot_n_elem: int,
    s_sub: float,
    k_sub: float,
    s_wall: float,
    k_wall: float,
    s_wall_end: float,
    k_wall_end: float,
    variation_level: float,
    n_trials: int
) -> List[Dict[str, npt.NDArray[np.float64]]]:
    """
    Intended only for generating paired friction arrays combinations with channel_parameter_list

    Outputs a list of dictionaries, each containing:
    - 'kinetic_mu_substrate_array': Kinetic friction coefficients for substrate [n_elem+1]
    - 'static_mu_substrate_array': Static friction coefficients for substrate [n_elem+1]
    - 'kinetic_mu_wall_array': Kinetic friction coefficients for wall [n_elem+1]
    - 'static_mu_wall_array': Static friction coefficients for wall [n_elem+1]
    """

    fric_array_list: List[Dict[str, npt.NDArray[np.float64]]] = []

    for i in range(n_trials):
        if s_sub - variation_level > 0:
            s_sub = _truncated_normal_sample_std1(
                loc=s_sub,
                scale=variation_level
            )
        else:
            s_sub = _truncated_normal_sample_std1(
                loc=s_sub,
                scale=s_sub * variation_level
            )
        
        if k_sub - variation_level > 0:
            k_sub = _truncated_normal_sample_std1(
                loc=k_sub,
                scale=variation_level
            )
        else:
            k_sub = _truncated_normal_sample_std1(
                loc=k_sub,
                scale=k_sub * variation_level
            )

        if s_wall - variation_level > 0:
            s_wall = _truncated_normal_sample_std1(
                loc=s_wall,
                scale=variation_level
            )
        else:
            s_wall = _truncated_normal_sample_std1(
                loc=s_wall,
                scale=s_wall * variation_level
            )
        
        if k_wall - variation_level > 0:
            k_wall = _truncated_normal_sample_std1(
                loc=k_wall,
                scale=variation_level
            )
        else:
            k_wall = _truncated_normal_sample_std1(
                loc=k_wall,
                scale=k_wall * variation_level
            )
        
        if s_wall_end - variation_level > 0:
            s_wall_end = _truncated_normal_sample_std1(
                loc=s_wall_end,
                scale=variation_level
            )
        else:
            s_wall_end = _truncated_normal_sample_std1(
                loc=s_wall_end,
                scale=s_wall_end * variation_level
            )

        if k_wall_end - variation_level > 0:
            k_wall_end = _truncated_normal_sample_std1(
                loc=k_wall_end,
                scale=variation_level
            )
        else:
            k_wall_end = _truncated_normal_sample_std1(
                loc=k_wall_end,
                scale=k_wall_end * variation_level
            )

        if k_sub > s_sub:
            k_sub = s_sub
        if k_wall > s_wall:
            k_wall = s_wall
        if k_wall_end > s_wall_end:
            k_wall_end = s_wall_end

        kinetic_mu_substrate_array = np.full((robot_n_elem + 1,), k_sub)
        static_mu_substrate_array = np.full((robot_n_elem + 1,), s_sub)
        kinetic_mu_wall_array = np.full((robot_n_elem + 1,), k_wall)
        static_mu_wall_array = np.full((robot_n_elem + 1,), s_wall)

        kinetic_mu_wall_array[:1] = k_wall_end
        kinetic_mu_wall_array[-1:] = k_wall_end
        static_mu_wall_array[:1] = s_wall_end
        static_mu_wall_array[-1:] = s_wall_end

        fric_array_list.append({
            'kinetic_mu_substrate_array': kinetic_mu_substrate_array,
            'static_mu_substrate_array': static_mu_substrate_array,
            'kinetic_mu_wall_array': kinetic_mu_wall_array,
            'static_mu_wall_array': static_mu_wall_array
        })

    return fric_array_list

def _wrapper_for_shape_evaluation_function(
    length: float,
    beam_width: float,
    beam_thickness: float,
    n_waveform: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_polygon: Polygon,
    channel_length: float,
    channel_x_mid: npt.NDArray[np.float64],
    channel_y_mid: npt.NDArray[np.float64],
    dist_threshold: float,
    robot_init_pos: npt.NDArray[np.float64],
    robot_init_origin: npt.NDArray[np.float64],
    robot_init_director: npt.NDArray[np.float64],
    B_field: float,
    B_frequency: float,
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    upper_bound_robot_length: float,
    n_elem: int,
    magnetization_direction: npt.NDArray[np.float64],
    variation_level=0.0,
    dynamic_fric_coeff=False,
    dynamic_fric_variation_level=0.0,
    backward=False,
    move_end_line_to=1.0,
    bifurcation=False,
    which_art='right',
    right_endline=0.0,
    left_endline=0.0
) -> np.float64:

    post_processing_dict, actual_final_time_in_ms, reached_which_endline = run_crawling_sim(
        length=length,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        n_waveform=n_waveform,
        beam_density=beam_density,
        modulus=modulus,
        magnetization_density=magnetization_density,
        channel_polygon=channel_polygon,
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos,
        rod_origin=robot_init_origin,
        rod_director=robot_init_director,
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        magnetization_direction=magnetization_direction,
        variation_level=variation_level,
        dynamic_fric_coeff=dynamic_fric_coeff,
        dynamic_fric_variation_level=dynamic_fric_variation_level,
        backward=backward,
        move_end_line_to=move_end_line_to,
        bifurcation=bifurcation,
        right_endline=right_endline,
        left_endline=left_endline,
        n_elem=n_elem
    )

    positions_over_time = np.array(post_processing_dict["position"])

    # Catch unexpected compute error from sim here
    if np.isnan(positions_over_time).any() or positions_over_time.shape[0] == 0 or actual_final_time_in_ms == np.float64(0.0):
        return np.float64(0.0)
    
    actual_final_time_in_s = actual_final_time_in_ms * MS_TO_SECONDS
    traveled_distance_midline = _travel_distance_along_midline(channel_x_mid, channel_y_mid, positions_over_time, move_end_line_to)

    speed = traveled_distance_midline / actual_final_time_in_s

    to_be_crawled_distance = 2 * upper_bound_robot_length + 0.01

    distance_left = to_be_crawled_distance - traveled_distance_midline
    
    # If the robot can't reach 0.5 * channel_length which is the tightest width, penalize severely.
    # Get how much distance left for (0.6 * channel_length - 0.5 * channel_length).
    # If the distance_left exceeds 0.1 * channel_length, it means the robot is to the left (outside) of the desired
    # transition location.
    if distance_left > 0.1 * channel_length:
        penalization_factor = 5.0
    else:
        penalization_factor = 2.0

    P = np.exp(-penalization_factor * distance_left / to_be_crawled_distance)

    score = P * speed

    # positions_over_time = np.array(post_processing_dict["position"])

    # # Catch unexpected compute error from sim here
    # if np.isnan(positions_over_time).any() or positions_over_time.shape[0] == 0 or actual_final_time_in_ms == np.float64(0.0):
    #     return np.float64(0.0)
    
    # actual_final_time_in_s = actual_final_time_in_ms * MS_TO_SECONDS
    # traveled_distance_midline = _travel_distance_along_midline(channel_x_mid, channel_y_mid, positions_over_time, move_end_line_to)

    # speed = traveled_distance_midline / actual_final_time_in_s

    # to_be_crawled_distance = 60.0

    # distance_left = to_be_crawled_distance - traveled_distance_midline
    
    # # If the robot can't reach 0.5 * channel_length which is the tightest width, penalize severely.
    # # Get how much distance left for (0.6 * channel_length - 0.5 * channel_length).
    # # If the distance_left exceeds 0.1 * channel_length, it means the robot is to the left (outside) of the desired
    # # transition location.
    # # if distance_left > 0.1 * channel_length:
    # #     penalization_factor = 5.0
    # # else:
    # #     penalization_factor = 2.0

    # P = np.exp(-3.0 * distance_left / to_be_crawled_distance)

    # score = P * speed

    # This is the time it took for the robot to crawl in sim
    return score

    # return score

def shape_objective_function(
    BO_parameters: Dict[str, float | str],
    fixed_parameters: Dict[str, float | int | List[Polygon | float | npt.NDArray[np.float64]]]
) -> Dict[str, np.float64]:
    
    required_bo_params = {
        'robot_length', 'robot_thickness', 'robot_n_waveform',
        'robot_material'
    }
    missing_params = required_bo_params - set(BO_parameters.keys())
    if missing_params:
        raise ValueError(f"Missing required BO parameters: {missing_params}")
    
    robot_length: float = BO_parameters['robot_length']
    robot_thickness: float = BO_parameters['robot_thickness']
    robot_n_waveform: float = BO_parameters['robot_n_waveform']

    robot_width: float = fixed_parameters['robot_width']
    robot_density: float = fixed_parameters['robot_density']
    robot_modulus: float = fixed_parameters['robot_modulus']
    robot_magnetization_density: float = fixed_parameters['robot_magnetization_density']
    target_dl: float = fixed_parameters['target_dl']
    cfl: float = fixed_parameters['cfl']
    B_field: float = fixed_parameters['B_field']
    B_frequency: float = fixed_parameters['B_frequency']
    backward: bool = fixed_parameters['backward']
    final_time: float = fixed_parameters['final_time']
    save_directory: str = fixed_parameters['save_directory']
    channel_type: str = fixed_parameters['channel_type']
    # patient_name: str = fixed_parameters['patient_name']
    # image_name: str = fixed_parameters['image_name']
    trial_num: str = fixed_parameters['trial_num']
    # segment_str_list: List[str] = fixed_parameters['segment_str_list']
    variation_level: float = fixed_parameters['variation_level']
    dynamic_fric_coeff: bool = fixed_parameters['dynamic_fric_coeff']
    dynamic_fric_variation_level: float = fixed_parameters['dynamic_fric_variation_level']
    n_channels: int = fixed_parameters['n_channels']
    n_trials: int = fixed_parameters['n_trials']
    upper_bound_robot_length: float = fixed_parameters['upper_bound_robot_length']
    end_line_percent: float = fixed_parameters['end_line_percent']
    offset_factor: float = fixed_parameters['offset_factor']
    precise_offset: float = fixed_parameters['precise_offset']

    channel_length = fixed_parameters['channel_length']

    channel_polygon_list: List[Polygon] = fixed_parameters['channel_polygon_list']
    x_midline_list: List[npt.NDArray[np.float64]] = fixed_parameters['x_midline_list']
    y_midline_list: List[npt.NDArray[np.float64]] = fixed_parameters['y_midline_list']
    left_bank_list: List[npt.NDArray[np.float64]] = fixed_parameters['left_bank_list']
    right_bank_list: List[npt.NDArray[np.float64]] = fixed_parameters['right_bank_list']
    total_length_list: List[float] = fixed_parameters['total_length_list']

    # List containing n_channels of Dictionary
    channel_parameter_list = []

    static_mu_wall: float = fixed_parameters['static_mu_wall']
    kinetic_mu_wall: float = fixed_parameters['kinetic_mu_wall']
    static_mu_wall_end: float = fixed_parameters['static_mu_wall_end']
    kinetic_mu_wall_end: float = fixed_parameters['kinetic_mu_wall_end']
    static_mu_substrate: float = fixed_parameters['static_mu_substrate']
    kinetic_mu_substrate: float = fixed_parameters['kinetic_mu_substrate']
    k_mag_den: float = fixed_parameters['k_mag_den']
    k_modulus: float = fixed_parameters['k_modulus']

    robot_magnetization_density *= k_mag_den
    robot_modulus *= k_modulus

    n_elem = int(robot_length / target_dl)
    if n_elem * target_dl < robot_length:
        n_elem += 1
    else:
        n_elem = n_elem

    dl = robot_length / n_elem
    dt = round_down(np.float64(cfl * dl / np.sqrt(robot_modulus / robot_density)))

    dist_threshold = 0.01

    # All channel list have 5 elements.
    start_point_offset = (0.6 * channel_length - 0.5 * robot_length - 2 * upper_bound_robot_length) / channel_length

    for i in range(len(channel_polygon_list)):
        channel_seg_midline = np.column_stack((x_midline_list[i], y_midline_list[i]))

        robot_init_pos = cu.generate_fiber_in_segment(channel_seg_midline,
                                                    L_fiber=robot_length,
                                                    offset_factor=start_point_offset,
                                                    N_fiber=n_elem)
        
        robot_init_director = cu.compute_directors_from_positions(robot_init_pos)
        robot_init_origin = robot_init_pos[:, 0]

        channel_parameter_list.append(
        {
            "channel_polygon": channel_polygon_list[i],
            "channel_length": total_length_list[i],
            "channel_x_mid": x_midline_list[i],
            "channel_y_mid": y_midline_list[i],
            "robot_init_pos": robot_init_pos,
            "robot_init_origin": robot_init_origin,
            "robot_init_director": robot_init_director
        }
        )

        # cu.plot_channel_and_fiber(channel_polygon_list[i], channel_seg_midline, robot_init_pos, 'test', end_line_percent)

    friction_coefficient_arrays_list = _fric_var_1std_for_sim_wrapper(
        n_elem,
        static_mu_substrate,
        kinetic_mu_substrate,
        static_mu_wall,
        kinetic_mu_wall,
        static_mu_wall_end,
        kinetic_mu_wall_end,
        variation_level,
        n_trials
    )

    n_elem = robot_init_pos.shape[1] - 1
    magnetization_direction = np.zeros((3, n_elem))

    magnetization_angles = np.linspace(0, 2 * np.pi * robot_n_waveform, n_elem)
    magnetization_direction[0, :] = np.cos(magnetization_angles)
    magnetization_direction[1, :] = np.sin(magnetization_angles)

    # Cartesian Product (sets in Mathematics)
    ParameterPair = namedtuple('ParameterPair', ['channel_params', 'friction_params'])
    all_pairs = [ParameterPair(channel, friction) for channel, friction in itertools.product(channel_parameter_list, friction_coefficient_arrays_list)]

    joblib_results = Parallel(n_jobs=-1)(delayed(_wrapper_for_shape_evaluation_function)(
        length=robot_length,
        beam_width=robot_width,
        beam_thickness=robot_thickness,
        n_waveform=robot_n_waveform,
        beam_density=robot_density,
        modulus=robot_modulus,
        magnetization_density=robot_magnetization_density,
        channel_polygon=pair.channel_params['channel_polygon'],
        channel_length=pair.channel_params['channel_length'],
        channel_x_mid=pair.channel_params['channel_x_mid'],
        channel_y_mid=pair.channel_params['channel_y_mid'],
        dist_threshold=dist_threshold,
        robot_init_pos=pair.channel_params['robot_init_pos'],
        robot_init_origin=pair.channel_params['robot_init_origin'],
        robot_init_director=pair.channel_params['robot_init_director'],
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=pair.friction_params['kinetic_mu_substrate_array'],
        static_mu_substrate_array=pair.friction_params['static_mu_substrate_array'],
        kinetic_mu_wall_array=pair.friction_params['kinetic_mu_wall_array'],
        static_mu_wall_array=pair.friction_params['static_mu_wall_array'],
        dt=dt,
        final_time=final_time,
        upper_bound_robot_length=upper_bound_robot_length,
        n_elem=n_elem,
        magnetization_direction=magnetization_direction,
        variation_level=variation_level,
        dynamic_fric_coeff=dynamic_fric_coeff,
        dynamic_fric_variation_level=dynamic_fric_variation_level,
        backward=backward,
        move_end_line_to=end_line_percent
        ) for pair in all_pairs
    )

    # Reshape the 1D array of joblib_results into (n_channels, n_trials)
    scores_matrix = np.array(joblib_results).reshape(
        len(channel_parameter_list),
        len(friction_coefficient_arrays_list)
    )

    # Sum across channels of a column
    sums_per_friction = np.sum(scores_matrix, axis=0)

    # Average across the columns
    average = np.mean(sums_per_friction)

    # Where the true values likely lie in for a given friction condition but
    # across diverse channel geometry
    sem = np.std(sums_per_friction, ddof=1) / np.sqrt(len(sums_per_friction))

    objective = {
        "averageSpeed": average,
        "sem": sem
    }

    # # Sum across channels of a column
    # sums_per_friction = np.sum(scores_matrix, axis=0)

    # # Average across the columns
    # average = np.mean(sums_per_friction)

    # # Where the true values likely lie in for a given friction condition but
    # # across diverse channel geometry
    # sem = np.std(sums_per_friction, ddof=1) / np.sqrt(len(sums_per_friction))

    # average_across_friction = np.mean(scores_matrix, axis=1)
    # summed_time = np.sum(average_across_friction)

    # objective = {
    #     "summedTime": summed_time
    # }

    with open(os.path.join(save_directory, 'channel_results', channel_type,\
                            trial_num,\
                            'debug.txt'), 'a') as file:
        file.write(f"Objectives:\n")
        file.write(f"{objective}\n")
        file.close()

    return objective

def _wrapper_for_sinu_evaluation_function(
    length: float,
    beam_width: float,
    beam_thickness: float,
    n_waveform: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_polygon: Polygon,
    channel_length: float,
    channel_x_mid: npt.NDArray[np.float64],
    channel_y_mid: npt.NDArray[np.float64],
    dist_threshold: float,
    robot_init_pos: npt.NDArray[np.float64],
    robot_init_origin: npt.NDArray[np.float64],
    robot_init_director: npt.NDArray[np.float64],
    B_field: float,
    B_frequency: float,
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    upper_bound_robot_length: float,
    variation_level=0.0,
    dynamic_fric_coeff=False,
    dynamic_fric_variation_level=0.0,
    backward=False,
    move_end_line_to=1.0
) -> np.float64:

    post_processing_dict, actual_final_time_in_ms, reached_end_line = run_crawling_sim(
        length=length,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        n_waveform=n_waveform,
        beam_density=beam_density,
        modulus=modulus,
        magnetization_density=magnetization_density,
        channel_polygon=channel_polygon,
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos,
        rod_origin=robot_init_origin,
        rod_director=robot_init_director,
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        variation_level=variation_level,
        dynamic_fric_coeff=dynamic_fric_coeff,
        dynamic_fric_variation_level=dynamic_fric_variation_level,
        backward=backward,
        move_end_line_to=move_end_line_to
    )

    positions_over_time = np.array(post_processing_dict["position"])

    # Catch unexpected compute error from sim here
    if np.isnan(positions_over_time).any() or positions_over_time.shape[0] == 0 or actual_final_time_in_ms == np.float64(0.0):
        return np.float64(0.0)
    
    actual_final_time_in_s = actual_final_time_in_ms * MS_TO_SECONDS
    traveled_distance_midline = _travel_distance_along_midline(channel_x_mid, channel_y_mid, positions_over_time, move_end_line_to)

    speed = traveled_distance_midline / actual_final_time_in_s

    # Intentional that it isn't a exact match of 60mm
    to_be_crawled_distance = 60.0

    distance_left = to_be_crawled_distance - traveled_distance_midline
    
    # If the robot can't reach 0.5 * channel_length which is the tightest width, penalize severely.
    # Get how much distance left for (0.6 * channel_length - 0.5 * channel_length).
    # If the distance_left exceeds 0.1 * channel_length, it means the robot is to the left (outside) of the desired
    # transition location.
    # if distance_left > 0.1 * channel_length:
    #     penalization_factor = 5.0
    # else:
    #     penalization_factor = 2.0

    P = np.exp(-3.0 * distance_left / to_be_crawled_distance)

    score = P * speed

    return score

def sinu_objective_function(
    BO_parameters: Dict[str, float | str],
    fixed_parameters: Dict[str, float | int | List[Polygon | float | npt.NDArray[np.float64]]]
) -> Dict[str, np.float64]:
    
    required_bo_params = {
        'robot_length', 'robot_thickness', 'robot_n_waveform',
        'robot_material'
    }
    missing_params = required_bo_params - set(BO_parameters.keys())
    if missing_params:
        raise ValueError(f"Missing required BO parameters: {missing_params}")
    
    robot_length: float = BO_parameters['robot_length']
    robot_thickness: float = BO_parameters['robot_thickness']
    robot_n_waveform: float = BO_parameters['robot_n_waveform']
    robot_material: str = BO_parameters['robot_material']

    robot_width: float = fixed_parameters['robot_width']
    robot_density: float = fixed_parameters['robot_density']
    robot_modulus: float = fixed_parameters['robot_modulus']
    robot_magnetization_density: float = fixed_parameters['robot_magnetization_density']
    target_dl: float = fixed_parameters['target_dl']
    cfl: float = fixed_parameters['cfl']
    B_field: float = fixed_parameters['B_field']
    B_frequency: float = fixed_parameters['B_frequency']
    backward: bool = fixed_parameters['backward']
    final_time: float = fixed_parameters['final_time']
    save_directory: str = fixed_parameters['save_directory']
    channel_type: str = fixed_parameters['channel_type']
    trial_num: str = fixed_parameters['trial_num']
    variation_level: float = fixed_parameters['variation_level']
    dynamic_fric_coeff: bool = fixed_parameters['dynamic_fric_coeff']
    dynamic_fric_variation_level: float = fixed_parameters['dynamic_fric_variation_level']
    n_channels: int = fixed_parameters['n_channels']
    n_trials: int = fixed_parameters['n_trials']
    move_end_line_to: float = fixed_parameters['move_end_line_to']
    upper_bound_robot_length: float = fixed_parameters['upper_bound_robot_length']
    channel_length: float = fixed_parameters['channel_length']

    channel_polygon_list: List[Polygon] = fixed_parameters['channel_polygon_list']
    x_midline_list: List[npt.NDArray[np.float64]] = fixed_parameters['x_midline_list']
    y_midline_list: List[npt.NDArray[np.float64]] = fixed_parameters['y_midline_list']
    left_bank_list: List[npt.NDArray[np.float64]] = fixed_parameters['left_bank_list']
    right_bank_list: List[npt.NDArray[np.float64]] = fixed_parameters['right_bank_list']
    total_length_list: List[float] = fixed_parameters['total_length_list']

    # List containing n_channels of Dictionary
    channel_parameter_list = []

    static_mu_wall: float = fixed_parameters['static_mu_wall']
    kinetic_mu_wall: float = fixed_parameters['kinetic_mu_wall']
    static_mu_wall_end: float = fixed_parameters['static_mu_wall_end']
    kinetic_mu_wall_end: float = fixed_parameters['kinetic_mu_wall_end']
    static_mu_substrate: float = fixed_parameters['static_mu_substrate']
    kinetic_mu_substrate: float = fixed_parameters['kinetic_mu_substrate']
    k_mag_den: float = fixed_parameters['k_mag_den']
    k_modulus: float = fixed_parameters['k_modulus']

    robot_magnetization_density *= k_mag_den
    robot_modulus *= k_modulus

    n_elem = int(robot_length / target_dl)
    if n_elem * target_dl < robot_length:
        n_elem += 1
    else:
        n_elem = n_elem

    dl = robot_length / n_elem
    dt = round_down(np.float64(cfl * dl / np.sqrt(robot_modulus / robot_density)))

    dist_threshold = 0.01

    start_point_offset = (move_end_line_to * channel_length - 0.5 * robot_length - 2 * 20.0) / channel_length

    for i in range(len(channel_polygon_list)):
        channel_seg_midline = np.column_stack((x_midline_list[i], y_midline_list[i]))

        robot_init_pos = cu.generate_fiber_in_segment(channel_seg_midline,
                                                    L_fiber=robot_length,
                                                    offset_factor=start_point_offset,
                                                    N_fiber=n_elem)
        
        robot_init_director = cu.compute_directors_from_positions(robot_init_pos)
        robot_init_origin = robot_init_pos[:, 0]

        # cu.plot_channel_and_fiber(channel_polygon_list[i], channel_seg_midline, robot_init_pos, "Sinusoidal", move_end_line_to)

        channel_parameter_list.append(
        {
            "channel_polygon": channel_polygon_list[i],
            "channel_length": total_length_list[i],
            "channel_x_mid": x_midline_list[i],
            "channel_y_mid": y_midline_list[i],
            "robot_init_pos": robot_init_pos,
            "robot_init_origin": robot_init_origin,
            "robot_init_director": robot_init_director
        }
        )

    friction_coefficient_arrays_list = _fric_var_1std_for_sim_wrapper(
        n_elem,
        static_mu_substrate,
        kinetic_mu_substrate,
        static_mu_wall,
        kinetic_mu_wall,
        static_mu_wall_end,
        kinetic_mu_wall_end,
        variation_level,
        n_trials
    )

    # Cartesian Product (sets in Mathematics)
    ParameterPair = namedtuple('ParameterPair', ['channel_params', 'friction_params'])
    all_pairs = [
        ParameterPair(channel, friction)
        for channel, friction in itertools.product(
            channel_parameter_list,
            friction_coefficient_arrays_list
        )
    ]

    joblib_results = Parallel(n_jobs=-1)(delayed(_wrapper_for_sinu_evaluation_function)(
        length=robot_length,
        beam_width=robot_width,
        beam_thickness=robot_thickness,
        n_waveform=robot_n_waveform,
        beam_density=robot_density,
        modulus=robot_modulus,
        magnetization_density=robot_magnetization_density,
        channel_polygon=pair.channel_params['channel_polygon'],
        channel_length=pair.channel_params['channel_length'],
        channel_x_mid=pair.channel_params['channel_x_mid'],
        channel_y_mid=pair.channel_params['channel_y_mid'],
        dist_threshold=dist_threshold,
        robot_init_pos=pair.channel_params['robot_init_pos'],
        robot_init_origin=pair.channel_params['robot_init_origin'],
        robot_init_director=pair.channel_params['robot_init_director'],
        B_field=B_field,
        B_frequency=B_frequency,
        kinetic_mu_substrate_array=pair.friction_params['kinetic_mu_substrate_array'],
        static_mu_substrate_array=pair.friction_params['static_mu_substrate_array'],
        kinetic_mu_wall_array=pair.friction_params['kinetic_mu_wall_array'],
        static_mu_wall_array=pair.friction_params['static_mu_wall_array'],
        dt=dt,
        final_time=final_time,
        upper_bound_robot_length=20.0,
        variation_level=variation_level,
        dynamic_fric_coeff=dynamic_fric_coeff,
        dynamic_fric_variation_level=dynamic_fric_variation_level,
        backward=backward,
        move_end_line_to=move_end_line_to
        ) for pair in all_pairs
    )

    # Reshape the 1D array of joblib_results into (n_channels, n_trials)
    scores_matrix = np.array(joblib_results).reshape(
        len(channel_parameter_list),
        len(friction_coefficient_arrays_list)
    )

    # Sum across channels of a column
    sums_per_friction = np.sum(scores_matrix, axis=0)

    # Average across the columns
    average = np.mean(sums_per_friction)

    # Where the true values likely lie in for a given friction condition but
    # across diverse channel geometry
    sem = np.std(sums_per_friction, ddof=1) / np.sqrt(len(sums_per_friction))

    objective = {
        "averageScore": average,
        "sem": sem
    }

    with open(os.path.join(fixed_parameters['save_directory'], 'channel_results', fixed_parameters['channel_type'], fixed_parameters['trial_num'], 'debug.txt'), 'a') as file:
        file.write(f"objective:\n")
        file.write(f"{objective}\n")
        file.close()

    return objective