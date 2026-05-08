import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from scipy.interpolate import splprep, BSpline, make_splprep
from scipy.optimize import least_squares
from typing import Tuple, Dict, List, Any
from elastica import *
from magneto_pyelastica import *
from interaction_channel import ChannelInteraction
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Parallel, delayed
import sys
import os

MIN_DT_RATIO = 0.1
DEFAULT_BOUNDARY_STIFFNESS = 0.1
DEFAULT_CONTACT_DAMPING = 1e-10
DEFAULT_VELOCITY_THRESHOLD = 1e-2
DEFAULT_DAMPING_CONSTANT = 0.1
FORCE_TO_MICRO_NEWTON = 1e6
DEFAULT_STEP_SKIP = 500
DT_REDUCTION_FACTOR = 0.2
PLOT_MARGIN = 5
ANIMATION_FPS = 15
ANIMATION_BITRATE = 1800
ANIMATION_INTERVAL = 100
optimization_count = 0

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

    def make_callback(self, system: Any, time: float, current_step: int) -> None:
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

def run_bending_sim(
    length: float,
    beam_width: float,
    beam_thickness: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_polygon: Polygon,
    dist_threshold: float,
    robot_init_pos: npt.NDArray[np.float64],
    rod_origin: npt.NDArray[np.float64],
    rod_director: npt.NDArray[np.float64],
    B_field: float,
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    variation_level: float = 0.0,
) -> Tuple[Dict[str, List], float]:
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
        
    Returns:
        Tuple:
            - Dictionary with simulation data (energies, positions, forces, etc.)
            - Actual final simulation time [ms]
            
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If simulation fails to converge
    """
    
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
            
            magnetic_field = B_field * 1e-6
            
            n_elem = robot_init_pos.shape[1] - 1
            magnetization_direction = np.zeros((3, n_elem))
            
            magnetization_direction[0, :] = 1

            magnetic_beam_sim = MagneticBeamSimulator()
            
            magnetic_rod = CosseratRod.straight_rod(
                n_elem,
                start=rod_origin.reshape(3),
                direction=np.array([0.0, 1.0, 0.0]),
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

            magnetic_beam_sim.constrain(magnetic_rod).using(
                OneEndFixedBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,)
            )
            
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
            )
            
            magnetic_field_object = ConstantMagneticField(
                magnetic_field_amplitude=magnetic_field * np.array([0.0, 1.0, 0.0]),
                ramp_interval=1e2 * dt,
                start_time=0,
                end_time=100 * final_time
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
            
            actual_final_time_in_ms = mag_den_stepping(
                timestepper, magnetic_beam_sim, final_time,
                n_steps=total_steps, progress_bar=True
            )
            simulation_successful = True
            
        except Exception as e:
            last_exception = e
            dt -= original_dt * DT_REDUCTION_FACTOR
    
    if not simulation_successful:
        error_msg = (
            f"Simulation failed to converge after reducing dt to {dt:.4f} "
            f"({MIN_DT_RATIO*100}% of original {original_dt:.4f}). "
            f"Last error: {str(last_exception)}"
        )
        raise RuntimeError(error_msg)
    
    return post_processing_dict, actual_final_time_in_ms

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
    ax.set_aspect('equal')

    # Plot channel boundary
    x_poly, y_poly = channel_polygon.exterior.xy
    ax.plot(x_poly, y_poly, 'k-', linewidth=2, label="Channel")

    # Extract dimensions
    n_frames, _, n_points = positions_over_time.shape

    # Initialize robot visualization
    initial_pos = positions_over_time[0, :2, :]  # x, y coordinates
    fiber_line, = ax.plot(
        initial_pos[0, :], initial_pos[1, :], 
        'r-', linewidth=2, label="Robot"
    )

    # Set plot limits with margin
    ax.set_xlim(min(x_poly) - PLOT_MARGIN, max(x_poly) + PLOT_MARGIN)
    ax.set_ylim(min(y_poly) - PLOT_MARGIN, max(y_poly) + PLOT_MARGIN)
    ax.legend(loc='upper right')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(movie_name) # 15/7/2025 CHANGE: Directly use movie_name

    def update_animation(frame: int) -> Tuple:
        """Update function for animation."""
        current_pos = positions_over_time[frame, :2, :]
        fiber_line.set_data(current_pos[0, :], current_pos[1, :])
        return (fiber_line,)

    # Create animation
    animation_obj = animation.FuncAnimation(
        fig, update_animation, frames=n_frames,
        interval=ANIMATION_INTERVAL, blit=True, repeat=True
    )

    # Save animation
    try:
        writer = animation.writers['ffmpeg'](
            fps=ANIMATION_FPS, 
            metadata={'artist': 'Magnetic Beam Robot Simulation'},
            bitrate=ANIMATION_BITRATE
        )
        animation_obj.save(os.path.join(save_directory, movie_name), writer=writer)
    except Exception as e:
        print(f"Warning: Could not save animation '{movie_name}': {e}")
    finally:
        plt.close(fig)

def objective_function(
    magnetization_density: float,
    OPTIMIZATION_SCALE: float,
    length: float,
    beam_width: float,
    beam_thickness: float,
    beam_density: float,
    modulus: float,
    channel_list: List[Polygon],
    dist_threshold: float,
    robot_init_pos_list: List[npt.NDArray[np.float64]],
    robot_init_origin_list: List[npt.NDArray[np.float64]],
    robot_init_director_list: List[npt.NDArray[np.float64]],
    B_field_list: List[float],
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    exp_bend_angle_array: npt.NDArray[np.float64],
    dt: float,
    final_time: float,
    variation_level: float = 0.0
):
    global optimization_count
    magnetization_density = float(magnetization_density * OPTIMIZATION_SCALE)

    joblib_results = Parallel(n_jobs=-1)(delayed(run_bending_sim)(
        length=length,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        beam_density=beam_density,
        modulus=modulus,
        magnetization_density=magnetization_density,
        channel_polygon=channel_list[i],
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos_list[i],
        rod_origin=robot_init_origin_list[i],
        rod_director=robot_init_director_list[i],
        B_field=B_field_list[i],
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        variation_level=variation_level
        ) for i in range(len(B_field_list)))

    post_processing_tuple_dict, _ = zip(*joblib_results)
    post_processing_tuple_dict: tuple[Dict]

    initial_frame_pos_list = []
    final_frame_pos_list = []

    for i in range(len(post_processing_tuple_dict)):
        initial_frame_pos_list.append(post_processing_tuple_dict[i]["position"][0])
        final_frame_pos_list.append(post_processing_tuple_dict[i]["position"][-1])

    # shape(num_mT_sims, xy, n_nodes) with the z dropped due to 2D-plane bending
    initial_frame_pos_array: npt.NDArray[np.float64] = np.array(initial_frame_pos_list)
    final_frame_pos_array: npt.NDArray[np.float64] = np.array(final_frame_pos_list)
    initial_frame_pos_array = initial_frame_pos_array[:, :2, :]
    final_frame_pos_array = final_frame_pos_array[:, :2, :]

    # Get the vector (normalized) of the undeformed (reference) configuration
    # Get the tangent vector (normalized) of the tip
    # Get the bend angle = arccos(tangent_tip_vec dot ref_vec)

    x_ref_diffs = initial_frame_pos_array[:, 0, -1] - initial_frame_pos_array[:, 0, 0]
    y_ref_diffs = initial_frame_pos_array[:, 1, -1] - initial_frame_pos_array[:, 1, 0]

    # Shape(num_mT_sims, xy).
    # xy vectors normalized corresponding to each mT_sim
    ref_vectors = np.stack((x_ref_diffs, y_ref_diffs), axis=1)
    ref_vectors: npt.NDArray[np.float64] = ref_vectors / np.linalg.norm(ref_vectors, axis=1, keepdims=True)

    tip_tangent_vector_list = []

    for i in range(final_frame_pos_array.shape[0]):
        spline, _ = make_splprep([final_frame_pos_array[i, 0, :], final_frame_pos_array[i, 1, :]], s=2)
        spline_derivative = spline.derivative()
        # Shape (xy,) 1D array
        tip_tangent_vector = spline_derivative(1.0)
        tangent_norm = np.linalg.norm(tip_tangent_vector)

        if tangent_norm > 0:
            tip_tangent_vector = tip_tangent_vector / tangent_norm
        
        tip_tangent_vector_list.append(tip_tangent_vector)

    # Shape(num_mT_sims, xy)
    tip_tangent_vectors: npt.NDArray[np.float64] = np.array(tip_tangent_vector_list)
    
    dot_products = np.clip(np.sum(tip_tangent_vectors * ref_vectors, axis=1), -1.0, 1.0)

    # Bend angle definition of counter-clockwise positive from y-axis
    sim_bend_angle_array: npt.NDArray[np.float64] = np.arccos(dot_products)

    relative_error = (sim_bend_angle_array - exp_bend_angle_array) / exp_bend_angle_array

    print("------------------------------------------------")
    print("")
    print(f"Iteration {optimization_count} relative_error (sim - exp) / exp:")
    print(relative_error)
    print("Simulated bend angles (degree, actual cal in radian):")
    print(np.degrees(sim_bend_angle_array))
    print("Magnetization density:")
    print(f"{magnetization_density} A/m")
    print("")

    optimization_count += 1

    return relative_error

def wrapper_mag_den(
    length: float,
    beam_width: float,
    beam_thickness: float,
    beam_density: float,
    modulus: float,
    magnetization_density: float,
    channel_list: List[Polygon],
    dist_threshold: float,
    robot_init_pos_list: List[npt.NDArray[np.float64]],
    robot_init_origin_list: List[npt.NDArray[np.float64]],
    robot_init_director_list: List[npt.NDArray[np.float64]],
    B_field_list: List[float],
    kinetic_mu_substrate_array: npt.NDArray[np.float64],
    static_mu_substrate_array: npt.NDArray[np.float64],
    kinetic_mu_wall_array: npt.NDArray[np.float64],
    static_mu_wall_array: npt.NDArray[np.float64],
    exp_bend_angle_array: npt.NDArray[np.float64],
    OPTIMIZATION_SCALE: float,
    log_path: str,
    dt: float,
    final_time: float,
    movie_names_list: List[str],
    variation_level: float = 0.0,
    should_optimize=True
):
    len_b_list = len(B_field_list)
    len_channel_list = len(channel_list)
    len_robot_init_pos_list = len(robot_init_pos_list)
    len_robot_init_origin_list = len(robot_init_origin_list)
    len_robot_init_director_list = len(robot_init_director_list)
    len_movie_names_list = len(movie_names_list)

    if len_channel_list != len_b_list:
        raise ValueError(f"Channel list length mismatch. b_list {len_b_list}. channel_list {len_channel_list}.")
    if len_robot_init_pos_list != len_b_list:
        raise ValueError(f"Robot init pos list length mismatch. b_list {len_b_list}. robot_init_pos_list {len_robot_init_pos_list}.")
    if len_robot_init_origin_list != len_b_list:
        raise ValueError(f"Robot init origin list length mismatch. b_list {len_b_list}. robot_init_origin_list {len_robot_init_origin_list}.")
    if len_robot_init_director_list != len_b_list:
        raise ValueError(f"Robot init director list length mismatch. b_list {len_b_list}. robot_init_director_list {len_robot_init_director_list}.")
    if len_movie_names_list != len_b_list:
        raise ValueError(f"Movie names list length mismatch. b_list {len_b_list}. movie_names_list {len_movie_names_list}.")

    del len_b_list, len_channel_list, len_robot_init_pos_list, len_robot_init_origin_list, len_robot_init_director_list, len_movie_names_list

    if should_optimize:
        print("Optimization enabled.")
        solution = least_squares(
            fun=objective_function,
            x0=magnetization_density / OPTIMIZATION_SCALE,
            args=(
                OPTIMIZATION_SCALE,
                length,
                beam_width,
                beam_thickness,
                beam_density,
                modulus,
                channel_list,
                dist_threshold,
                robot_init_pos_list,
                robot_init_origin_list,
                robot_init_director_list,
                B_field_list,
                kinetic_mu_substrate_array,
                static_mu_substrate_array,
                kinetic_mu_wall_array,
                static_mu_wall_array,
                exp_bend_angle_array,
                dt,
                final_time,
                variation_level),
            ftol = 1e-8,
            xtol = 1e-8,
            gtol = 1e-8,
            bounds=(0, np.inf),
            max_nfev=200
        )

        optim_mag_den = solution.x[0] * OPTIMIZATION_SCALE

        if not solution.success:
            print("Error minimization failed! Reason:")
            print(f"{solution.message}\n")
            print(f"Best estimate of mag_den: {optim_mag_den}")
            print("Best relative error:")
            print(f"{solution.fun}\n")
            print(f"Number of function evaluations: {solution.nfev}")

            with open(log_path, 'a') as file:
                file.write("Error minimization failed! Reason:\n")
                file.write(f"{solution.message}\n")
                file.write(f"Best estimate of mag_den: {optim_mag_den} A/m\n")
                file.write("Best relative error (sim - exp) / exp:\n")
                file.write(f"{solution.fun}\n")
                file.write(f"Number of function evaluations: {solution.nfev}\n\n")
                file.write("Parameters used.\n")
                file.write(f"magnetization_density: {magnetization_density}\n")
                file.write(f"OPTIMIZATION_SCALE: {OPTIMIZATION_SCALE}\n")
                file.write("\n")
                file.write("---------------------------------------------------------------------------")
                file.write("\n")
                file.close()

            print(f"Failed minimization appended to: {log_path}")
            sys.exit()

        print("Error minimization success!")
        print(f"Best estimate of mag_den: {optim_mag_den}")
        print("Best relative error:")
        print(f"{solution.fun}\n")
        print(f"Number of function evaluations: {solution.nfev}")

        print(f"Creating simulations based on optimal magnetization density value {optim_mag_den} A/m...")

        joblib_results = Parallel(n_jobs=-1)(delayed(run_bending_sim)(
            length=length,
            beam_width=beam_width,
            beam_thickness=beam_thickness,
            beam_density=beam_density,
            modulus=modulus,
            magnetization_density=optim_mag_den,
            channel_polygon=channel_list[i],
            dist_threshold=dist_threshold,
            robot_init_pos=robot_init_pos_list[i],
            rod_origin=robot_init_origin_list[i],
            rod_director=robot_init_director_list[i],
            B_field=B_field_list[i],
            kinetic_mu_substrate_array=kinetic_mu_substrate_array,
            static_mu_substrate_array=static_mu_substrate_array,
            kinetic_mu_wall_array=kinetic_mu_wall_array,
            static_mu_wall_array=static_mu_wall_array,
            dt=dt,
            final_time=final_time,
            variation_level=variation_level
            ) for i in range(len(B_field_list)))

        post_processing_tuple_dict, _ = zip(*joblib_results)
        post_processing_tuple_dict: tuple[Dict]

        initial_frame_pos_list = []
        final_frame_pos_list = []

        for i in range(len(post_processing_tuple_dict)):
            initial_frame_pos_list.append(post_processing_tuple_dict[i]["position"][0])
            final_frame_pos_list.append(post_processing_tuple_dict[i]["position"][-1])

        initial_frame_pos_array: npt.NDArray[np.float64] = np.array(initial_frame_pos_list)
        final_frame_pos_array: npt.NDArray[np.float64] = np.array(final_frame_pos_list)
        initial_frame_pos_array = initial_frame_pos_array[:, :2, :]
        final_frame_pos_array = final_frame_pos_array[:, :2, :]

        x_ref_diffs = initial_frame_pos_array[:, 0, -1] - initial_frame_pos_array[:, 0, 0]
        y_ref_diffs = initial_frame_pos_array[:, 1, -1] - initial_frame_pos_array[:, 1, 0]

        ref_vectors = np.stack((x_ref_diffs, y_ref_diffs), axis=1)
        ref_vectors: npt.NDArray[np.float64] = ref_vectors / np.linalg.norm(ref_vectors, axis=1, keepdims=True)

        tip_tangent_vector_list = []

        for i in range(final_frame_pos_array.shape[0]):
            spline, _ = make_splprep([final_frame_pos_array[i, 0, :], final_frame_pos_array[i, 1, :]], s=2)
            spline_derivative = spline.derivative()
            tip_tangent_vector = spline_derivative(1.0)
            tangent_norm = np.linalg.norm(tip_tangent_vector)

            if tangent_norm > 0:
                tip_tangent_vector = tip_tangent_vector / tangent_norm
            
            tip_tangent_vector_list.append(tip_tangent_vector)

            if i == 0:
                # Check curvature and then bending strain
                u_param = np.linspace(0, 1, final_frame_pos_array.shape[-1])

                # Shape (xy, N_points)
                dr_du = spline.derivative()
                # Shape (xy, N_points)
                d2r_du2 = dr_du.derivative()

                # Means (dr/du)|u=points_in_u_param, or derivative evaluated at u_param
                dr_du_value = dr_du(u_param)
                d2r_du2_value = d2r_du2(u_param)

                # Shape (N_points, xy)
                dr_du_value = dr_du_value.T
                d2r_du2_value = d2r_du2_value.T

                # dr_du x d2r_du2
                cross_product = np.cross(dr_du_value, d2r_du2_value)

                # |dr_du × d2r_du2|
                norm_cross = np.abs(cross_product)
                # |dr_du|
                norm_dr_du = np.linalg.norm(dr_du_value, axis=1)

                # Shape (N_points,)
                # kappa(u) = |dr_du x d2r_du2| / |dr_du|^3
                curvature: npt.NDArray[np.float64] = norm_cross / (norm_dr_du ** 3)

        tip_tangent_vectors: npt.NDArray[np.float64] = np.array(tip_tangent_vector_list)
        
        dot_products = np.clip(np.sum(tip_tangent_vectors * ref_vectors, axis=1), -1.0, 1.0)

        sim_bend_angle_array: npt.NDArray[np.float64] = np.arccos(dot_products)

        r_equivalent = (beam_width * beam_thickness ** 3 / (3 * np.pi)) ** 0.25

        with open(log_path, 'a') as file:
            file.write("Error minimization success!\n")
            file.write(f"Best estimate of mag_den: {optim_mag_den} A/m\n")
            file.write("Simulated bend angles in degree:\n")
            file.write(f"{np.degrees(sim_bend_angle_array)}\n")
            file.write("Best relative error (sim - exp) / exp:\n")
            file.write(f"{solution.fun}\n")
            file.write(f"Number of elements: {u_param.shape}\n")
            file.write(f"Curvature values:\n")
            file.write(f"{curvature}\n")
            file.write(f"Percentage of elements having bending strain >= 0.05: {((r_equivalent * curvature) >= 0.05).mean() * 100}\n")
            file.write(f"Bending strain in %:\n")
            file.write(f"{r_equivalent * curvature * 100}\n")
            file.write(f"Number of function evaluations: {solution.nfev}\n\n")
            file.write("Parameters used.\n")
            file.write(f"magnetization_density: {magnetization_density}\n")
            file.write(f"OPTIMIZATION_SCALE: {OPTIMIZATION_SCALE}\n")
            file.write("\n")
            file.write("---------------------------------------------------------------------------")
            file.write("\n")
            file.close()

        print(f"Successful minimization appended to: {log_path}")

        for i in range(len(post_processing_tuple_dict)):
            _create_simulation_movie(
                np.array(post_processing_tuple_dict[i]["position"]),
                channel_list[i],
                movie_names_list[i],
                os.path.dirname(log_path)
            )
        print("Program successfully completed.")
    else:
        print("Optimization disabled.")

        joblib_results = Parallel(n_jobs=-1)(delayed(run_bending_sim)(
        length=length,
        beam_width=beam_width,
        beam_thickness=beam_thickness,
        beam_density=beam_density,
        modulus=modulus,
        magnetization_density=magnetization_density,
        channel_polygon=channel_list[i],
        dist_threshold=dist_threshold,
        robot_init_pos=robot_init_pos_list[i],
        rod_origin=robot_init_origin_list[i],
        rod_director=robot_init_director_list[i],
        B_field=B_field_list[i],
        kinetic_mu_substrate_array=kinetic_mu_substrate_array,
        static_mu_substrate_array=static_mu_substrate_array,
        kinetic_mu_wall_array=kinetic_mu_wall_array,
        static_mu_wall_array=static_mu_wall_array,
        dt=dt,
        final_time=final_time,
        variation_level=variation_level
        ) for i in range(len(B_field_list)))

        post_processing_tuple_dict, _ = zip(*joblib_results)
        post_processing_tuple_dict: tuple[Dict]

        initial_frame_pos_list = []
        final_frame_pos_list = []

        for i in range(len(post_processing_tuple_dict)):
            initial_frame_pos_list.append(post_processing_tuple_dict[i]["position"][0])
            final_frame_pos_list.append(post_processing_tuple_dict[i]["position"][-1])

        initial_frame_pos_array: npt.NDArray[np.float64] = np.array(initial_frame_pos_list)
        final_frame_pos_array: npt.NDArray[np.float64] = np.array(final_frame_pos_list)
        initial_frame_pos_array = initial_frame_pos_array[:, :2, :]
        final_frame_pos_array = final_frame_pos_array[:, :2, :]

        x_ref_diffs = initial_frame_pos_array[:, 0, -1] - initial_frame_pos_array[:, 0, 0]
        y_ref_diffs = initial_frame_pos_array[:, 1, -1] - initial_frame_pos_array[:, 1, 0]

        ref_vectors = np.stack((x_ref_diffs, y_ref_diffs), axis=1)
        ref_vectors: npt.NDArray[np.float64] = ref_vectors / np.linalg.norm(ref_vectors, axis=1, keepdims=True)

        tip_tangent_vector_list = []

        for i in range(final_frame_pos_array.shape[0]):
            tck, _ = splprep([final_frame_pos_array[i, 0, :], final_frame_pos_array[i, 1, :]], s=2)
            spline = BSpline(tck[0], np.column_stack(tck[1]), tck[2])
            spline_derivative = spline.derivative()
            tip_tangent_vector = spline_derivative(1.0)
            tangent_norm = np.linalg.norm(tip_tangent_vector)

            if tangent_norm > 0:
                tip_tangent_vector = tip_tangent_vector / tangent_norm
            
            tip_tangent_vector_list.append(tip_tangent_vector)

        tip_tangent_vectors: npt.NDArray[np.float64] = np.array(tip_tangent_vector_list)

        dot_products = np.clip(np.sum(tip_tangent_vectors * ref_vectors, axis=1), -1.0, 1.0)

        sim_bend_angle_array: npt.NDArray[np.float64] = np.arccos(dot_products)
        sim_bend_angle_array = np.degrees(sim_bend_angle_array)

        with open(log_path, 'a') as file:
            file.write("Optimization disabled.\n")
            for i in range(sim_bend_angle_array.shape[0]):
                file.write(f"{str(B_field_list[i])}mT tip tangent to undeformed configuration angle (degree): {sim_bend_angle_array[i]}\n")
            file.write("\n")
            file.write("---------------------------------------------------------------------------")
            file.write("\n")
        file.close()

        print(f"Computed bend angles appended: {log_path}")

        for i in range(len(post_processing_tuple_dict)):
            _create_simulation_movie(
                np.array(post_processing_tuple_dict[i]["position"]),
                channel_list[i],
                movie_names_list[i],
                os.path.dirname(log_path)
            )