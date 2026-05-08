"""
For channel_preprocess_utility.py
"""

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from scipy.spatial import KDTree
from typing import List, Tuple

def compute_arc_length(points: npt.NDArray[np.float64]):
    """
    Compute the cumulative arc-length along a sequence of 2D points.

    Parameters:
      points: ndarray of shape (M, 2)

    Returns:
      arc: 1D ndarray of length M containing the cumulative arc-length.
    """
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc: npt.NDArray[np.float64] = np.concatenate(([0], np.cumsum(dists)))
    return arc

def compute_width(
    midline: npt.NDArray[np.float64],
    left_bank: npt.NDArray[np.float64],
    right_bank: npt.NDArray[np.float64]
) -> Tuple[List[np.float64], List[int], List[int], np.float64, np.float64, List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Cast a normal vector line from the midline.
    Find one point on both left and right boundary closest to this casted line.
    The distance between the two points is the width.
    This technique is much better than assuming each left and right boundary array point-pairs
    have their width intersect the midline perpendicularly
    Both left_bank and right_bank are (N,2) arrays.

    Returns:
        The found width w.r.t. the midline as a 1D List (N_midline_points,)
        The found closest points on the left and right boundary to the casted line
        minimum width as np.float64
        maximum width as np.float64
        indices Lists to index the points on the original left and right boundary array
    """
    width_list = []
    right_idx_list = []
    left_idx_list = []
    left_tree = KDTree(left_bank)
    right_tree = KDTree(right_bank)

    tangent = np.gradient(midline, axis=0)
    tangent_norm = np.linalg.norm(tangent)
    unit_tangent = tangent / tangent_norm
    normal = np.column_stack((-unit_tangent[:, 1], unit_tangent[:, 0]))

    multi = np.linspace(0, 20, int(2e3))

    print("Computing widths. This may take a while...")

    for i in range(midline.shape[0]):
        mid_point = midline[i, :]

        # Re-use previously computed normal and multi
        perpendicular_casted_line = mid_point + normal[i, :] * multi[:, np.newaxis]
        
        # KDTree way of finding nearest neighbor
        left_dists, left_indices = left_tree.query(perpendicular_casted_line)
        right_dists, right_indices = right_tree.query(perpendicular_casted_line)
        
        # Find which point on casted line gives the minimum distance
        left_min_idx = np.argmin(left_dists)
        right_min_idx = np.argmin(right_dists)
        
        left_closest_point_idx = left_indices[left_min_idx]
        right_closest_point_idx = right_indices[right_min_idx]

        # Compute width
        width_vec = left_bank[left_closest_point_idx, :] - right_bank[right_closest_point_idx, :]
        width = np.linalg.norm(width_vec)

        width_list.append(width)

        # These two list are storing the int value corresponding to smooth left or right boundary
        right_idx_list.append(right_closest_point_idx)
        left_idx_list.append(left_closest_point_idx)
    print("Width compute finished.")

    min_width: np.float64 = min(width_list)
    min_indices_in_width_list = [i for i, width in enumerate(width_list) if width == min_width]

    max_width: np.float64 = max(width_list)
    max_indices_in_width_list = [i for i, width in enumerate(width_list) if width == max_width]

    min_indices_in_smooth_right_boundary = [right_idx_list[idx_map] for _, idx_map in enumerate(min_indices_in_width_list)]
    min_indices_in_smooth_left_boundary = [left_idx_list[idx_map] for _, idx_map in enumerate(min_indices_in_width_list)]

    max_indices_in_smooth_right_boundary = [right_idx_list[idx_map] for _, idx_map in enumerate(max_indices_in_width_list)]
    max_indices_in_smooth_left_boundary = [left_idx_list[idx_map] for _, idx_map in enumerate(max_indices_in_width_list)]

    return width_list, right_idx_list, left_idx_list, min_width, max_width, min_indices_in_width_list, max_indices_in_width_list, min_indices_in_smooth_right_boundary, min_indices_in_smooth_left_boundary, max_indices_in_smooth_right_boundary, max_indices_in_smooth_left_boundary

def compute_curvature(midline: npt.NDArray[np.float64],
                     arc: npt.NDArray[np.float64]):
    """
    Approximate curvature along a 2D midline using finite differences.
    curvature = |dtheta/ds|, where theta = arctan2(dy, dx).
    
    Parameters:
      midline: ndarray of shape (M,2)
      arc: 1D ndarray of cumulative arc-length values corresponding to midline.
    
    Returns:
      curvature: 1D ndarray of shape (N_points,) curvature values along the midline.
    """
    dx = np.gradient(midline[:, 0], arc, edge_order=2)
    dy = np.gradient(midline[:, 1], arc, edge_order=2)
    theta = np.unwrap(np.arctan2(dy, dx))
    curvature: npt.NDArray[np.float64] = np.abs(np.gradient(theta, arc, edge_order=2))
    return curvature

def get_centered_window_indices(arc: npt.NDArray[np.float64],
                                center_index: int,
                                L_seg: float) -> tuple[int, int]:
    """
    Given the arc-length array and a center index, return indices corresponding to a
    symmetric window of arc-length L_seg centered at arc[center_index].
    If the desired window extends beyond the available arc, the window is "snapped"
    to the beginning or end so that the extracted segment has (approximately) length L_seg.

    Arguments
    ---------
    arc:
        1D array of shape (N_points,)
    center_index:
        index corresponding to the middle of the segemented arc from the overall arc array.
    L_seg:
        Length of the segment.

    Returns
    -------
    i_start:
        int index for the starting point of the segment from the overall arc array.
    i_end:
        int index for the ending point of the segment from the overall arc array.
    """
    total_channel_length = arc[-1]
    N_points = arc.shape[0]
    desired_start: np.float64 = arc[center_index] - L_seg / 2
    desired_end: np.float64 = arc[center_index] + L_seg / 2

    if desired_start < arc[0]:
        i_start = 0
        # Force window to have arc length L_seg
        i_end: int = np.searchsorted(arc, arc[0] + L_seg, side='right')
        return i_start, i_end

    if desired_end > total_channel_length:
        i_end = N_points - 1
        i_start: int = np.searchsorted(arc, total_channel_length - L_seg, side='left')
        return i_start, i_end

    # Normal case: extract indices for desired_start and desired_end.
    i_start = np.searchsorted(arc, desired_start, side='left')
    i_end = np.searchsorted(arc, desired_end, side='right')
    return i_start, i_end

def get_end_closed_lines(seg_poly: Polygon):
    """
    Given a segment polygon (seg_poly), extract the two end closed lines.

    Assumes that seg_poly.exterior.coords was built as:
       left_seg (in order) concatenated with right_seg (in reverse order)
    so that the first half corresponds to the left bank and the second half
    to the right bank.

    Returns:
       start_line: ndarray of shape (2, 2)
           The line connecting the first left bank point to the last right bank point.
       end_line: ndarray of shape (2, 2)
           The line connecting the last left bank point to the first right bank point.
    """
    # Convert the exterior coordinates to a numpy array and remove the closing duplicate.
    coords = np.array(seg_poly.exterior.coords)[:-1]
    # Assume the first half corresponds to the left bank, second half to the right bank.
    n = coords.shape[0] // 2
    left_bank_coords = coords[:n]
    right_bank_coords = coords[n:]

    # The "start end line" connects the first point of the left bank with the last point of the right bank.
    start_line: npt.NDArray[np.float64] = np.array([left_bank_coords[0], right_bank_coords[-1]])
    # The "end end line" connects the last point of the left bank with the first point of the right bank.
    end_line: npt.NDArray[np.float64] = np.array([left_bank_coords[-1], right_bank_coords[0]])

    return start_line, end_line

def round_down(dt: np.float64) -> np.float64:
    """
    Round down a time step (dt) to the closest rational number of decimal places.
    """
    decimals = max(0, int(-np.floor(np.log10(dt))) + 1)
    factor = 10 ** decimals
    return np.floor(dt * factor) / factor