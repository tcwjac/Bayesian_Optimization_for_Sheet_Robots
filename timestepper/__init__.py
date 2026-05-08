__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""


import numpy as np
import math
from tqdm import tqdm
from elastica.timestepper.symplectic_steppers import (
    SymplecticStepperTag,
    PositionVerlet,
    PEFRL,
)
from elastica.timestepper.explicit_steppers import (
    ExplicitStepperTag,
    RungeKutta4,
    EulerForward,
)
from elastica.modules.base_system import BaseSystemCollection
from shapely.geometry import Polygon, LineString
from typing import Tuple
import numpy.typing as npt
from collections import deque

# TODO: Both extend_stepper_interface and integrate should be in separate file.
# __init__ is probably not an ideal place to have these scripts.
def extend_stepper_interface(Stepper, System):
    from elastica.utils import extend_instance
    from elastica.systems import is_system_a_collection

    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    is_this_system_a_collection = is_system_a_collection(System)

    """
    # Stateful steppers are no more used so remove them
    ConcreteStepper = (
        Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
    )
    """
    ConcreteStepper = Stepper

    if type(ConcreteStepper.Tag) == SymplecticStepperTag:
        from elastica.timestepper.symplectic_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            SymplecticStepperMethods as StepperMethodCollector,
        )
    elif type(ConcreteStepper.Tag) == ExplicitStepperTag:
        from elastica.timestepper.explicit_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            ExplicitStepperMethods as StepperMethodCollector,
        )
    # elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
    #    return  # hacky fix for now. remove HybridSteppers in a future version.
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                ConcreteStepper.__class__.__name__
            )
        )

    stepper_methods = StepperMethodCollector(ConcreteStepper)
    do_step_method = (
        _SystemCollectionStepper.do_step
        if is_this_system_a_collection
        else _SystemInstanceStepper.do_step
    )
    return do_step_method, stepper_methods.step_methods()


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(
    StatefulStepper,
    System,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
    **kwargs,
):
    """

    Parameters
    ----------
    StatefulStepper :
        Stepper algorithm to use.
    System :
        The elastica-system to simulate.
    final_time : float
        Total simulation time. The timestep is determined by final_time / n_steps.
    n_steps : int
        Number of steps for the simulation. (default: 1000)
    restart_time : float
        The timestamp of the first integration step. (default: 0.0)
    progress_bar : bool
        Toggle the tqdm progress bar. (default: True)
    """
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    # Extend the stepper's interface after introspecting the properties
    # of the system. If system is a collection of small systems (whose
    # states cannot be aggregated), then stepper now loops over the system
    # state
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(float(final_time) / n_steps)
    time = restart_time

    for i in tqdm(range(n_steps), disable=(not progress_bar)):
        time = do_step(StatefulStepper, stages_and_updates, System, time, dt)

    print("Final time of simulation is : ", time)
    return time

def check_robot_touch_end_line(fiber, end_line, threshold):
    """
    Original collision detection function (kept for backward compatibility).
    For better performance, use FastCollisionDetector class instead.
    """
    # Use only x and y coordinates.
    pts = fiber[:2, :].T if fiber.shape[0] >= 2 else fiber.T

    seg_start = np.asarray(end_line[0])
    seg_end = np.asarray(end_line[1])
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0:
        # Degenerate segment: use distance to seg_start.
        dists = np.linalg.norm(pts - seg_start, axis=1)
    else:
        # Vectorized projection
        pt_vecs = pts - seg_start  # (N, 2)
        t = np.dot(pt_vecs, seg_vec) / seg_len_sq  # (N,)
        t = np.clip(t, 0, 1)
        projections = seg_start + np.outer(t, seg_vec)
        dists = np.linalg.norm(pts - projections, axis=1)

    min_distance = dists.min()
    return (min_distance < threshold), min_distance


def get_end_closed_lines(seg_poly: Polygon,
                         backward=False,
                         move_end_line_to=1.0):
    """
    Given a segment polygon (seg_poly), extract the two end closed lines.

    Assumes that seg_poly.exterior.coords was built as:
       left_seg (in order) concatenated with right_seg (in reverse order)
    so that the first half corresponds to the left bank and the second half
    to the right bank.

    backward means crawl from right to left which the start-end line shifts accordingly.

    move_end_line_to means where the end line should be positioned along the\\
    channel length's percentage [0, 1]

    Returns:
       start_line: ndarray of shape (2, 2). The line connecting the first left bank point to the last right bank point.

       end_line: ndarray of shape (2, 2). The line connecting the last left bank point to the first right bank point.
    """
    # Convert the exterior coordinates to a numpy array and remove the closing duplicate.
    coords = np.array(seg_poly.exterior.coords)[:-1]

    # Assume the first half corresponds to the left bank, second half to the right bank.
    n = coords.shape[0] // 2
    left_bank_coords = coords[:n]

    # Be careful that this is right bank but reversed.
    right_bank_reversed_coords = coords[n:]

    left_bank = LineString(left_bank_coords)

    # Un-reverse it so that both left and right bank are aligned along channel direction.
    right_bank = LineString(right_bank_reversed_coords[::-1])

    left_bank_target_dist = move_end_line_to * left_bank.length
    right_bank_target_dist = move_end_line_to * right_bank.length

    left_target_pt = left_bank.interpolate(left_bank_target_dist)
    right_target_pt = right_bank.interpolate(right_bank_target_dist)

    # The two lines of code are defined for forward crawling.
    # The "start end line" connects the first point of the left bank with the last point of the right bank.
    start_line = np.array([left_bank_coords[0], right_bank_reversed_coords[-1]])
    # The "end end line" connects the last point of the left bank with the first point of the right bank.
    end_line = np.array([[left_target_pt.x, left_target_pt.y], [right_target_pt.x, right_target_pt.y]])

    if backward == True:
        temp = end_line
        end_line = start_line
        start_line = temp

    return start_line, end_line

class FastCollisionDetector:
    """
    Optimized collision detector that precomputes line parameters and uses
    fast squared distance calculations with early termination.
    """
    
    def __init__(self, end_line: npt.NDArray[np.float64], threshold=0.5, check_subset_ratio=0.3, move_end_line_to=1.0):
        """
        Initialize collision detector with precomputed line parameters.
        
        Parameters
        ----------
        end_line : ndarray of shape (2, 2)
            Line defined by two 2D points. It isn't an actual line. It's the array containing the two end points to define a line.
        threshold : float
            Collision detection threshold
        check_subset_ratio : float
            Fraction of robot points to check (0.3 = front 30% of robot)
        """
        self.seg_start = np.asarray(end_line[0], dtype=np.float64)
        self.seg_end = np.asarray(end_line[1], dtype=np.float64)
        self.seg_vec = self.seg_end - self.seg_start
        self.seg_len_sq = np.dot(self.seg_vec, self.seg_vec)
        self.threshold_sq = threshold * threshold  # Use squared threshold
        self.threshold = threshold
        self.check_subset_ratio = check_subset_ratio
        
        # Handle degenerate line case
        self.is_degenerate = (self.seg_len_sq < 1e-12)

        self.move_end_line_to = move_end_line_to
        
    def check_collision_fast(self, fiber_positions: npt.NDArray[np.float64]) -> Tuple[bool, np.float64]:
        """
        Fast collision check with early termination and subset checking.
        
        Parameters
        ----------
        fiber_positions : ndarray of shape (3, N) or (2, N)
            Robot point positions
            
        Returns
        -------
        bool
            True if collision detected
        float
            Minimum distance found (only computed if collision detected).
            Actually the min distance to end line is returned regardless of touching the end line
        """
        # Extract x,y coordinates directly (avoid transpose)
        x_coords = fiber_positions[0, :]
        y_coords = fiber_positions[1, :]
        n_points = len(x_coords)
        
        if self.move_end_line_to < 1.0:
            index_length = fiber_positions.shape[1]
            mid_node_index = index_length // 2
            fiber_positions = fiber_positions[:, mid_node_index]

            # Scalar
            x_coords = fiber_positions[0]
            y_coords = fiber_positions[1]
            n_points = 1

        # Check subset of points (typically front 30% of robot)
        elif self.check_subset_ratio < 1.0:
            subset_size = max(1, int(n_points * self.check_subset_ratio))
            # Check front points (assume robot moves forward)
            x_coords = x_coords[-subset_size:]  # Last points = front of robot
            y_coords = y_coords[-subset_size:]
        
        if self.is_degenerate:
            # Degenerate line: check distance to point
            dx = x_coords - self.seg_start[0]
            dy = y_coords - self.seg_start[1]
            dist_sq = dx*dx + dy*dy
        else:
            # Vectorized point-to-line distance calculation
            # For each point, find closest point (robot, denoted as x/y_coords) on line segment
            # seg_start is one of the end point
            dx = x_coords - self.seg_start[0]
            dy = y_coords - self.seg_start[1]
            
            # Project onto line: t = dot(pt_vec, seg_vec) / seg_len_sq
            t = (dx * self.seg_vec[0] + dy * self.seg_vec[1]) / self.seg_len_sq
            t = np.clip(t, 0.0, 1.0)  # Clamp to line segment
            
            # Closest points on line segment
            closest_x = self.seg_start[0] + t * self.seg_vec[0]
            closest_y = self.seg_start[1] + t * self.seg_vec[1]
            
            # Squared distances
            dx_closest = x_coords - closest_x
            dy_closest = y_coords - closest_y
            dist_sq = dx_closest*dx_closest + dy_closest*dy_closest
        
        # Find minimum squared distance
        min_dist_sq = np.min(dist_sq)
        
        # Early return for collision check
        if min_dist_sq < self.threshold_sq:
            return True, np.sqrt(min_dist_sq)
        else:
            return False, np.sqrt(min_dist_sq)

    def check_collision_full(self, fiber_positions: npt.NDArray[np.float64]):
        """
        Full collision check (all robot points) for final verification.
        """
        # Temporarily disable subset checking
        original_ratio = self.check_subset_ratio
        self.check_subset_ratio = 1.0
        result = self.check_collision_fast(fiber_positions)
        self.check_subset_ratio = original_ratio
        return result

def integrate_polygon_optimized(
    StatefulStepper: PositionVerlet,
    System: BaseSystemCollection,
    final_time: float,
    channel_polygon: Polygon,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
    collision_threshold: float = 0.5,
    check_subset_ratio: float = 0.3,
    system_index: int = 0,
    backward=False,
    move_end_line_to=1.0,
    bifur=False,
    right_endline=0.0,
    left_endline=0.0,
    **kwargs,
) -> Tuple[np.float64, str]:
    """
    If you use move_end_line_to with value < 1.0, the mid node of the robot is used for checking if the end line\\
    is touched. If the value is 1.0, then the forward tip of the robot is used for the end line collision check.

    Optimized integration with fast collision detection.
    
    Performance improvements:
    - Precomputed line parameters (no repeated calculations)
    - Squared distance comparisons (avoid sqrt until needed)
    - Subset checking (only check front portion of robot)
    - Direct memory access (no unnecessary copies)
    - Early termination
    
    Parameters
    ----------
    check_subset_ratio : float
        Fraction of robot points to check for collision (0.3 = front 30%)
        Smaller values = faster but potentially less accurate collision detection
    """
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"
    assert 0.0 < check_subset_ratio <= 1.0, "check_subset_ratio must be in (0, 1]"

    # Setup integration
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)
    dt = np.float64(float(final_time) / n_steps)
    time = restart_time

    if bifur == False:
        # Initialize optimized collision detector
        start_line, end_line = get_end_closed_lines(channel_polygon, backward, move_end_line_to)
        collision_detector = FastCollisionDetector(
            end_line, collision_threshold, check_subset_ratio, move_end_line_to
        )

        # Integration loop with optimized collision detection
        for i in tqdm(range(n_steps), disable=(not progress_bar)):
            time = do_step(StatefulStepper, stages_and_updates, System, time, dt)
            
            # Direct access to position data (no copy)
            # Shape (xyz, n_nodes)
            position_collection: npt.NDArray[np.float64] = System._systems[system_index].position_collection

            # Fast collision check
            reached_end_line, min_distance = collision_detector.check_collision_fast(position_collection)
            
            if reached_end_line:
                print(f"End line (move_end_line_to {move_end_line_to}) reached at time {time:.3f}! Minimum absolute distance away from the real end line: {min_distance:.3f}mm")
                break

        return time, 'yes'
    
    else:
        right_collision_detector = FastCollisionDetector(
            right_endline, collision_threshold, check_subset_ratio, move_end_line_to
        )
        left_collision_detector = FastCollisionDetector(
            left_endline, collision_threshold, check_subset_ratio, move_end_line_to
        )
        for i in tqdm(range(n_steps), disable=(not progress_bar)):
            time = do_step(StatefulStepper, stages_and_updates, System, time, dt)
            
            position_collection: npt.NDArray[np.float64] = System._systems[system_index].position_collection

            reached_right, min_distance = right_collision_detector.check_collision_fast(position_collection)
            reached_left, min_distance = left_collision_detector.check_collision_fast(position_collection)
            
            # Intentional to not raise an error here.
            # Let the program continues so that a video can be generated for debugging
            if reached_right and reached_left:
                print("Left and right endlines reached simultaneously!")
                return time, 'both'

            if reached_right:
                print(f"Reached right endline at time {time}ms!")
                return time, 'right'

            if reached_left:
                print(f"Reached left endline at time {time}ms!")
                return time, 'left'

        return time, 'neither'

def mag_den_stepping(
    StatefulStepper: PositionVerlet,
    System: BaseSystemCollection,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
    system_index: int = 0,
    **kwargs,
):
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    # Setup stepping
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)
    dt = np.float64(float(final_time) / n_steps)
    time = restart_time

    buffer_size = 100
    position_buffer = deque(maxlen=buffer_size)

    # Stepping loop
    for i in tqdm(range(n_steps), disable=(not progress_bar)):
        time = do_step(StatefulStepper, stages_and_updates, System, time, dt)
        
        # Shape (xy, n_nodes)
        position: npt.NDArray[np.float64] = System._systems[system_index].position_collection[:2, :].copy()
        position_buffer.append(position)

        # Moving average check to account for oscillatory deflection.
        # For a static deflection due to magnetic field, it shouldn't take long to stabilize.
        # So a 0.06 factor multiplication.
        if i >= int(math.log2(n_steps)) and len(position_buffer) == buffer_size:
            average_position = np.mean(np.stack(position_buffer), axis=0)
            relative_change = np.abs((position - average_position) / (average_position + 1e-12))

            if np.all(relative_change < 1e-9):
                print(f"Reached stable deformation. Terminating simulation...")
                break

    return time