"""
This file is intended to be gone through line-by-line carefully.
It's a bit long, but it can make or break everything if you don't go through the artery image to numpy representation program
and modify things you don't like.
"""

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import os
import sys
from bo_sheet.channel_preprocess_utility import create_smooth_curve, find_closest_idx_on_boundary, create_centered_figure
from bo_sheet.curve_compute_utils import compute_width

patient_name = "zeroFour"
image_name = "zeroFour_right"

parent_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'GitHub', 'Bayesian_Optimization_for_Sheet_Robots', \
                              'code', '3_optim_shape')

coronary_dir = os.path.join(parent_dir, 'channel_params', 'coronary_artery', patient_name)

pixel_to_mm = 0.258390625
N_points = 750
# channel_min_width = 0.5
# channel_max_width = 1.7

img = cv2.imread(os.path.join(coronary_dir, f"{image_name}.png"), cv2.IMREAD_GRAYSCALE)

smoothed_img = cv2.GaussianBlur(img, (5,5), 1)

fig = create_centered_figure()
plt.imshow(smoothed_img)
plt.title("Gaussian Smoothed Image")
plt.show()

_, binary = cv2.threshold(smoothed_img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Convert to (some_num_of_points, xy)
old_boundary_points: npt.NDArray[np.float64] = contours[0].reshape(-1, 2)

binary_bool = binary > 0

skeleton = skeletonize(binary_bool)

coords = np.column_stack(np.where(skeleton))
cost_array = np.where(skeleton, 1, 1e6)

endpoints = []
for y, x in coords:
    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
    if neighbors == 1:
        endpoints.append((y, x))

start, end = endpoints[0], endpoints[-1]

path, cost = route_through_array(cost_array, start, end, fully_connected=True)
ordered_midline = np.array(path)

# Convert to (456, xy)
old_midline_points = ordered_midline[:, ::-1].reshape(ordered_midline.shape[0], -1)

colors = plt.get_cmap('rainbow')

fig = create_centered_figure()
plt.imshow(img, cmap='gray')
for i in range(5):
    plt.plot(old_boundary_points[int(i * old_boundary_points.shape[0]/5) : int((i + 1) * old_boundary_points.shape[0]/5 + 1), 0],\
             old_boundary_points[int(i * old_boundary_points.shape[0]/5) : int((i + 1) * old_boundary_points.shape[0]/5 + 1), 1],\
             '.', color=colors(i / 4.0),\
             label=f"Boundary Array {int(i * old_boundary_points.shape[0]/5)}-{int((i + 1) * old_boundary_points.shape[0]/5)}")
    
for i in range(8):
    plt.plot(old_midline_points[int(i * old_midline_points.shape[0]/8) : int((i + 1) * old_boundary_points.shape[0]/8 + 1), 0],\
             old_midline_points[int(i * old_midline_points.shape[0]/8) : int((i + 1) * old_boundary_points.shape[0]/8 + 1), 1],\
             '.', color=colors(i / 7.0),\
             label=f"Midline Array {int(i * old_midline_points.shape[0]/8)}-{int((i + 1) * old_midline_points.shape[0]/8)}")
plt.legend()
plt.axis('off')
plt.title("Array Indices Correspondence")
plt.tight_layout()
plt.show()

# Convert to real mm dimension and x-axis positive rightward and y-axis positive upward
temp_midline_points = old_midline_points
temp_boundary_points = old_boundary_points

x_offset = temp_midline_points[0, 0]
y_offset = temp_midline_points[0, 1]

temp_midline_points[:, 0] -= x_offset
temp_midline_points[:, 1] -= y_offset
temp_midline_points[:, 1] *= -1

temp_boundary_points[:, 0] -= x_offset
temp_boundary_points[:, 1] -= y_offset
temp_boundary_points[:, 1] *= -1

new_midline_points = pixel_to_mm * temp_midline_points
new_boundary_points = pixel_to_mm * temp_boundary_points

midline = create_smooth_curve(new_midline_points, N_points)

midline_tangent: npt.NDArray[np.float64] = np.gradient(midline, axis=0)
unit_midline_tangent = midline_tangent / np.linalg.norm(midline_tangent)

start_point_reversed_tangent = -1 * unit_midline_tangent[0, :]

multi = np.linspace(0, 20, int(2e3))
casted_line = midline[0, :] + start_point_reversed_tangent * multi[:, np.newaxis]

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'k-', label="Midline")
plt.plot(casted_line[:, 0], casted_line[:, 1], 'g-', label="Casted Line for Start")
plt.title("Visual Check of Start Line Tangent Casting")
plt.legend()
plt.tight_layout()
plt.show()

# For each point on the boundary, compute the smallest distance to the casted_line and store the result as a dictionary
# After the for loop, sort the result so that we find the 2 smallest distance and which two indices they belong to from the boundary
dist_dict = {}
for i in range(0, 223):
    boundary_point = new_boundary_points[i]
    distances = np.linalg.norm(casted_line - boundary_point, axis=1)
    smallest_dist = np.min(distances)
    dist_dict[i] = smallest_dist

sorted_distances = sorted(dist_dict.items(), key=lambda x: x[1])
two_closest = sorted_distances[:2]

start_index_list = []
for idx, dist in two_closest:
    start_index_list.append(idx)

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'k-', label="Midline")
plt.plot(new_boundary_points[:, 0], new_boundary_points[:, 1], 'g.', label="Boundary Points")
plt.plot(new_boundary_points[start_index_list[0], 0], new_boundary_points[start_index_list[0], 1], 'r.', label=f"Index {start_index_list[0]}")
plt.plot(new_boundary_points[start_index_list[1], 0], new_boundary_points[start_index_list[1], 1], 'c.', label=f"Index {start_index_list[1]}")
plt.title("Start Indices for Separating Boundary Points Array")
plt.legend()
plt.tight_layout()
plt.show()

end_point_tangent = unit_midline_tangent[-1]
casted_line = midline[-1, :] + end_point_tangent * multi[:, np.newaxis]

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'k-', label="Midline")
plt.plot(casted_line[:, 0], casted_line[:, 1], 'g-', label="Casted Line for End")
plt.title("Visual Check of End Line Tangent Casting")
plt.legend()
plt.tight_layout()
plt.show()

dist_dict = {}
for i in range(446, 670):
    boundary_point = new_boundary_points[i]
    distances = np.linalg.norm(casted_line - boundary_point, axis=1)
    smallest_dist = np.min(distances)
    dist_dict[i] = smallest_dist

sorted_distances = sorted(dist_dict.items(), key=lambda x: x[1])
two_closest = sorted_distances[:2]

end_index_list = []

for idx, dist in two_closest:
    end_index_list.append(idx)

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'k-', label="Midline Points")
plt.plot(new_boundary_points[:, 0], new_boundary_points[:, 1], 'g.', label="Boundary Points")
plt.plot(new_boundary_points[end_index_list[0], 0], new_boundary_points[end_index_list[0], 1], 'r.', label=f"Index {end_index_list[0]}")
plt.plot(new_boundary_points[end_index_list[1], 0], new_boundary_points[end_index_list[1], 1], 'c.', label=f"Index {end_index_list[1]}")
plt.title("End Indices for Separating Boundary Points Array")
plt.legend()
plt.tight_layout()
plt.show()

# Now we have smooth midline and the start and end indices for both the left and right boundary.
# Then we partition the left and right boundary using midline normal vector line casting method
right_boundary = new_boundary_points[start_index_list[1] : end_index_list[0] + 1, :]

left_boundary = new_boundary_points[start_index_list[0] :: -1, :]
left_boundary = np.concatenate((left_boundary, new_boundary_points[-1 : end_index_list[1] + 1 : -1, :]))

right_boundary = create_smooth_curve(right_boundary, N_points)
left_boundary = create_smooth_curve(left_boundary, N_points)

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], '-', color='blue', label="Midline")
plt.plot(left_boundary[:, 0], left_boundary[:, 1], '-', color='red', label="Left Boundary")
plt.plot(right_boundary[:, 0], right_boundary[:, 1], '-', color='green', label="Right Boundary")
plt.title("Separated Boundary Points Array into Left and Right Boundary")
plt.legend()
plt.tight_layout()
plt.show()

midline_normal = np.column_stack((-unit_midline_tangent[:, 1], unit_midline_tangent[:, 0]))

# You must manually adjust the midline point indexing
trim_left_idx_start, trim_right_idx_start = find_closest_idx_on_boundary(True, midline[5, :], midline_normal[0, :], multi, left_boundary, right_boundary)
trim_left_idx_end, trim_right_idx_end = find_closest_idx_on_boundary(True, midline[-2, :], midline_normal[-1, :], multi, left_boundary, right_boundary)

midline = midline[5:-1, :]
midline = create_smooth_curve(midline, N_points)

left_boundary = left_boundary[trim_left_idx_start : trim_left_idx_end + 1, :]
right_boundary = right_boundary[trim_right_idx_start : trim_right_idx_end + 1, :]

left_boundary = create_smooth_curve(left_boundary, N_points)
right_boundary = create_smooth_curve(right_boundary, N_points)

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'b-', label="Midline")
plt.plot(left_boundary[:, 0], left_boundary[:, 1], 'r-', label="Left Boundary")
plt.plot(right_boundary[:, 0], right_boundary[:, 1], 'g-', label="Right Boundary")
plt.legend()
plt.title("Trimmed Start and End Boundary")
plt.tight_layout()
plt.show()

# Recommend putting sys.exit() so that you finalize the boundary and midline trimming first.
# sys.exit()

# Get width, closest idx on left and right boundary, min/max width values, ..., as names suggest
width_list, right_idx_list, left_idx_list, min_width, max_width, min_indices_in_width_list, max_indices_in_width_list, min_indices_in_right_boundary, min_indices_in_left_boundary, max_indices_in_right_boundary, max_indices_in_left_boundary = compute_width(midline, left_boundary, right_boundary)

# Unfortunately we don't have time for a repeated min or max width case
# If the following List have more than 1 element, manually select one
# print(min_indices_in_width_list)
# print(max_indices_in_width_list)
# print(min_indices_in_left_boundary)
# print(max_indices_in_left_boundary)
# print(min_indices_in_right_boundary)
# print(max_indices_in_right_boundary)

# sys.exit()

min_idx_in_width_list = min_indices_in_width_list[0]
max_idx_in_width_list = max_indices_in_width_list[0]
min_idx_in_left_boundary = min_indices_in_left_boundary[0]
max_idx_in_left_boundary = max_indices_in_left_boundary[0]
min_idx_in_right_boundary = min_indices_in_right_boundary[0]
max_idx_in_right_boundary = max_indices_in_right_boundary[0]

min_width_left_point = left_boundary[min_idx_in_left_boundary, :]
min_width_right_point = right_boundary[min_idx_in_right_boundary, :]
max_width_left_point = left_boundary[max_idx_in_left_boundary, :]
max_width_right_point = right_boundary[max_idx_in_right_boundary, :]

# Lines need to be in plt.plot([x1, x2], [y1, y2]) format
fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'b-', label="Midline")
plt.plot(left_boundary[:, 0], left_boundary[:, 1], 'r-', label="Left Boundary")
plt.plot(right_boundary[:, 0], right_boundary[:, 1], 'g-', label="Right Boundary")
plt.plot([max_width_left_point[0], max_width_right_point[0]], [max_width_left_point[1], max_width_right_point[1]], '-', color='orange', markersize=12, label=f"Max Width {max_width}mm")
plt.plot([min_width_left_point[0], min_width_right_point[0]], [min_width_left_point[1], min_width_right_point[1]], '-', color='cyan', markersize=12, label=f"Min Width {min_width}mm")
plt.legend()
plt.axis('on')
plt.title("Location of Maximum and Minimum Width with Numeric Values")
plt.show()

midline_length = np.cumsum(np.sqrt(np.sum(np.diff(midline, axis=0) ** 2, axis=1)))
midline_length = np.concatenate(([0], midline_length))

fig = create_centered_figure()
ax = plt.gca()
plt.plot(midline_length, width_list, 'b-', markersize=2)
plt.title("Contour Plot of Channel Width Along Midline")
plt.xlabel("Midline Length (mm)")
plt.ylabel("Width (mm)")
plt.axis('on')
plt.tight_layout()
plt.show()

scale = 2.0 / max_width
midline = scale * midline
left_boundary = scale * left_boundary
right_boundary = scale * right_boundary

width_list, right_idx_list, left_idx_list, min_width, max_width, min_indices_in_width_list, max_indices_in_width_list, min_indices_in_right_boundary, min_indices_in_left_boundary, max_indices_in_right_boundary, max_indices_in_left_boundary = compute_width(midline, left_boundary, right_boundary)

min_idx_in_width_list = min_indices_in_width_list[0]
max_idx_in_width_list = max_indices_in_width_list[0]
min_idx_in_left_boundary = min_indices_in_left_boundary[0]
max_idx_in_left_boundary = max_indices_in_left_boundary[0]
min_idx_in_right_boundary = min_indices_in_right_boundary[0]
max_idx_in_right_boundary = max_indices_in_right_boundary[0]

min_width_left_point = left_boundary[min_idx_in_left_boundary, :]
min_width_right_point = right_boundary[min_idx_in_right_boundary, :]
max_width_left_point = left_boundary[max_idx_in_left_boundary, :]
max_width_right_point = right_boundary[max_idx_in_right_boundary, :]

fig = create_centered_figure()
ax = plt.gca()
ax.set_aspect('equal')
plt.plot(midline[:, 0], midline[:, 1], 'b-', label="Midline")
plt.plot(left_boundary[:, 0], left_boundary[:, 1], 'r-', label="Left Boundary")
plt.plot(right_boundary[:, 0], right_boundary[:, 1], 'g-', label="Right Boundary")
plt.plot([max_width_left_point[0], max_width_right_point[0]], [max_width_left_point[1], max_width_right_point[1]], '-', color='orange', markersize=12, label=f"Max Width {max_width}mm")
plt.plot([min_width_left_point[0], min_width_right_point[0]], [min_width_left_point[1], min_width_right_point[1]], '-', color='cyan', markersize=12, label=f"Min Width {min_width}mm")
plt.legend()
plt.axis('on')
plt.title("Location of Scaled Maximum and Minimum Width with Numeric Values")
plt.show()

midline_length = np.cumsum(np.sqrt(np.sum(np.diff(midline, axis=0) ** 2, axis=1)))
midline_length = np.concatenate(([0], midline_length))
midline_length = midline_length[-1]

print(f"Scaled midline length: {midline_length}mm")

left_bank = (left_boundary[:, 0], left_boundary[:, 1])
right_bank = (right_boundary[:, 0], right_boundary[:, 1])

polygon_points = np.vstack([left_boundary, right_boundary[::-1]])

channel_info = {
    "x_midline": midline[:, 0],
    "y_midline": midline[:, 1],
    "left_bank": left_bank,
    "right_bank": right_bank,
    "polygon_points": polygon_points,
    "seg_mid": midline,
    "total_length": midline_length,
    "left_bound": left_boundary,
    "right_bound": right_boundary
}

input_result = input("Save channel information?")
if input_result == "y":
    np.savez(os.path.join(coronary_dir, f"{image_name}_channel_info"), **channel_info)
    print("Channel information saved to:")
    print(os.path.join(coronary_dir, f"{image_name}_channel_info.npz"))