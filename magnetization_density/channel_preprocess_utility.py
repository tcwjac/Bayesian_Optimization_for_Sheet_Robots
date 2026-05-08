import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from shapely.geometry import Polygon
from scipy.interpolate import interp1d

def gen_channel_for_mag_den(L_total: float, width=1.0, origin_offset=0.02):
    # Create simple rectangular channel with just 4 vertices
    half_length = 0.5 * L_total
    
    # Position channel so (0, 0) is at the desired offset along the midline
    x_center = -origin_offset * L_total + half_length
    
    # Create simple rectangle polygon
    channel_polygon = Polygon([
        (x_center - half_length, 0),
        (x_center + half_length, 0),
        (x_center + half_length, width),
        (x_center - half_length, width)
    ])
    
    # Create simple midline (just for compatibility)
    x_midline: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
    y_midline: npt.NDArray[np.float64] = np.array([0.0, 0.0])
    
    # Create simple banks
    left_x: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
    left_y: npt.NDArray[np.float64] = np.array([width, width])
    right_x: npt.NDArray[np.float64] = np.array([x_center - half_length, x_center + half_length])
    right_y: npt.NDArray[np.float64] = np.array([0, 0])
    
    left_bank = (left_x, left_y)
    right_bank = (right_x, right_y)
    
    return x_midline, y_midline, channel_polygon, left_bank, right_bank

def generate_fiber_in_segment(seg_midline: npt.NDArray[np.float64], L_fiber=5, offset_factor=0.0, N_fiber=50):
    """
    Generate a fiber path inside a channel segment.

    Parameters:
        seg_midline: ndarray of shape (M,2)
            Array of midline points for the segment. For straight channels, the shape is [2, 2], meaning start and end point
            xy position.
        L_fiber: float, optional
            Desired length of the fiber.
        offset_factor: float, optional
            Fraction of the total segment length at which the fiber begins.
        N_fiber: int, optional
            Number of points along the fiber.

    Returns:
        fiber: ndarray of shape (3, N_fiber), where 3 is x, y, and z rows
            Fiber coordinates with a zero z-coordinate.
    """
    # Compute cumulative arc-length along the segment midline.
    arc = np.cumsum(np.sqrt(np.sum(np.diff(seg_midline, axis=0) ** 2, axis=1)))
    arc = np.insert(arc, 0, 0)
    total_length = arc[-1]

    # Determine the start and end positions along the arc.
    start = offset_factor * total_length
    end = min(start + L_fiber, total_length)

    s_vals = np.linspace(start, end, N_fiber)

    # Create interpolation functions for x and y.
    interp_x = interp1d(arc, seg_midline[:, 0])
    interp_y = interp1d(arc, seg_midline[:, 1])

    fiber_x = interp_x(s_vals)
    fiber_y = interp_y(s_vals)

    # Return the fiber with z set to zero.
    # (xyz so shape 3, then N_fiber as columns). Row 0 is x. Row 1 is y
    fiber = np.vstack((fiber_x, fiber_y, np.zeros(N_fiber)))
    return fiber

def compute_directors_from_positions(robot_init_pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute director frames for a straight rod given the nodal positions.
    The out-of-plane normal is set to [0, 0, 1] by default.

    Parameters
    ----------
    robot_init_pos : np.ndarray
        Array of shape (3, N) containing the positions of N nodes.

    Returns
    -------
    directors : np.ndarray
        Array of shape (3, 3, n_elements) where n_elements = N-1.
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