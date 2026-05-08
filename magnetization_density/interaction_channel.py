"""
------Created at 2025.03.06 by Ziyu Ren.

The functions are updated for higher computational efficiency.
------Revised at 2025.06.08 by Ziyu Ren.

Revise this module to incoperate variations of friction coefficients during robot-channel interaction
------Revised at 2025.07.04 by Ziyu Ren.
"""

from shapely.geometry import Polygon
from elastica.external_forces import NoForces
from numba import njit, prange
import numpy as np
import numpy.typing as npt
import math
import elastica

@njit(cache=True)
def erf_inv(x: float) -> float:
    """Numba-compatible approximation of inverse error function"""
    # Winitzki approximation
    a = 0.147
    sign = 1 if x >= 0 else -1
    return sign * math.sqrt(math.sqrt((2/(math.pi*a) + 0.5*math.log(1-x**2))**2 - math.log(1-x**2)/a) - 
                      (2/(math.pi*a) + 0.5*math.log(1-x**2)))

@njit(cache=True)
def _truncated_normal_sample_std1(loc: float, scale: float) -> float:
    """
    Samples from N(loc, scale) truncated to [loc - scale, loc + scale]
    Equivalent to scipy.stats.truncnorm(-1, 1, loc, scale)
    """
    alpha = -1.0
    beta = 1.0
    
    # Calculate standard normal CDF values
    phi_alpha = 0.5 * (1 + math.erf(alpha / math.sqrt(2)))
    phi_beta = 0.5 * (1 + math.erf(beta / math.sqrt(2)))
    
    # Uniform sample in [phi_alpha, phi_beta]
    u = np.random.uniform(phi_alpha, phi_beta)
    
    # Inverse CDF transform
    sample = loc + scale * math.sqrt(2) * erf_inv(2 * u - 1)
    return sample

class ChannelInteraction(NoForces):
    def __init__(
            self,
            boundary_stiffness: float,
            contact_damping: float,
            dist_threshold: float,
            velocity_threshold: float,
            kinetic_mu_substrate_array: npt.NDArray[np.float64],
            static_mu_substrate_array: npt.NDArray[np.float64],
            kinetic_mu_wall_array: npt.NDArray[np.float64],
            static_mu_wall_array: npt.NDArray[np.float64],
            channel_polygon: Polygon,
            total_gravitational_force: float,
            variation_level=0.0,
    ):
        self.boundary_stiffness = boundary_stiffness
        self.contact_damping = contact_damping
        self.dist_threshold = dist_threshold
        self.velocity_threshold = velocity_threshold
        self.kinetic_mu_substrate_array = kinetic_mu_substrate_array
        self.static_mu_substrate_array = static_mu_substrate_array
        self.kinetic_mu_wall_array = kinetic_mu_wall_array
        self.static_mu_wall_array = static_mu_wall_array
        self.channel_polygon = channel_polygon
        self.total_gravitational_force = total_gravitational_force
        self.variation_level = variation_level
        
        self.boundary_coords = np.array(channel_polygon.exterior.coords)
        self.boundary_x = self.boundary_coords[:, 0]
        self.boundary_y = self.boundary_coords[:, 1]
        
        # Pre-compute point gravitational force (grav force spreaded over points)
        self.point_gravitational_force = total_gravitational_force / kinetic_mu_substrate_array.shape[0]

        if self.variation_level > 0.0:
            for i in prange(self.kinetic_mu_wall_array.shape[0]):
                if self.kinetic_mu_wall_array[i] - self.variation_level > 0:
                    self.kinetic_mu_wall_array[i] = _truncated_normal_sample_std1(
                        loc=self.kinetic_mu_wall_array[i],
                        scale=self.variation_level
                    )
                else:
                    self.kinetic_mu_wall_array[i] = _truncated_normal_sample_std1(
                        loc=self.kinetic_mu_wall_array[i],
                        scale=self.kinetic_mu_wall_array[i] * self.variation_level
                    )
                
                if self.static_mu_wall_array[i] - self.variation_level > 0:
                    self.static_mu_wall_array[i] = _truncated_normal_sample_std1(
                        loc=self.static_mu_wall_array[i],
                        scale=self.variation_level
                    )
                else:
                    self.static_mu_wall_array[i] = _truncated_normal_sample_std1(
                        loc=self.static_mu_wall_array[i],
                        scale=self.static_mu_wall_array[i] * self.variation_level
                    )

            for i in prange(self.kinetic_mu_substrate_array.shape[0]):
                if self.kinetic_mu_substrate_array[i] - self.variation_level > 0:
                    self.kinetic_mu_substrate_array[i] = _truncated_normal_sample_std1(
                        loc=self.kinetic_mu_substrate_array[i],
                        scale=self.variation_level
                    )
                else:
                    self.kinetic_mu_substrate_array[i] = _truncated_normal_sample_std1(
                        loc=self.kinetic_mu_substrate_array[i],
                        scale=self.kinetic_mu_substrate_array[i] * self.variation_level
                    )
                
                if self.static_mu_substrate_array[i] - self.variation_level > 0:
                    self.static_mu_substrate_array[i] = _truncated_normal_sample_std1(
                        loc=self.static_mu_substrate_array[i],
                        scale=self.variation_level
                    )
                else:
                    self.static_mu_substrate_array[i] = _truncated_normal_sample_std1(
                        loc=self.static_mu_substrate_array[i],
                        scale=self.static_mu_substrate_array[i] * self.variation_level
                    )

    def apply_forces(self, system: elastica.CosseratRod, time=0.0):
        return channel_force(
            self.boundary_stiffness,
            self.contact_damping,
            self.dist_threshold,
            self.velocity_threshold,
            self.kinetic_mu_substrate_array,
            self.static_mu_substrate_array,
            self.kinetic_mu_wall_array,
            self.static_mu_wall_array,
            self.boundary_x,
            self.boundary_y,
            self.point_gravitational_force,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )

@njit(cache=True)
def channel_force(
        boundary_stiffness: float,
        contact_damping: float,
        dist_threshold: float,
        velocity_threshold: float,
        kinetic_mu_substrate_array: npt.NDArray[np.float64],
        static_mu_substrate_array: npt.NDArray[np.float64],
        kinetic_mu_wall_array: npt.NDArray[np.float64],
        static_mu_wall_array: npt.NDArray[np.float64],
        boundary_x: npt.NDArray[np.float64],
        boundary_y: npt.NDArray[np.float64],
        point_gravitational_force: float,
        position_collection: npt.NDArray[np.float64],
        velocity_collection: npt.NDArray[np.float64],
        internal_forces: npt.NDArray[np.float64],
        external_forces: npt.NDArray[np.float64],
):
    """
    Compute all forces acting on the system including substrate friction, wall normal forces, and wall friction.
    This function orchestrates the computation of different force components.
    Internal and external torque are handled by PyElastica and Magneto Elastica.

    Args:
        position_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        velocity_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        internal_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements (segment between 2 nodes).
    """
    # Pre-compute total forces once
    # Shape (3, n_nodes)
    nodal_total_force_collection = _batch_vector_sum(internal_forces, external_forces)

    # Calculate substrate friction
    # Shape (3, n_nodes)
    substrate_friction_ndarray = compute_friction_force_substrate_array(
        position_collection,
        velocity_collection,
        nodal_total_force_collection,
        point_gravitational_force,
        kinetic_mu_substrate_array,
        static_mu_substrate_array,
        velocity_threshold
    )

    update_nodal_force(substrate_friction_ndarray, external_forces)

    # Calculate normal contact force with channel walls
    normal_force_ndarray = compute_normal_force_array_fast(
        position_collection,
        boundary_x,
        boundary_y,
        dist_threshold,
        boundary_stiffness,
        nodal_total_force_collection,
        velocity_collection,
        contact_damping
    )

    update_nodal_force(normal_force_ndarray, external_forces)

    # Calculate wall friction
    wall_friction_array = compute_friction_force_wall_array(
        position_collection,
        velocity_collection,
        nodal_total_force_collection,
        normal_force_ndarray,
        kinetic_mu_wall_array,
        static_mu_wall_array,
        velocity_threshold
    )

    update_nodal_force(wall_friction_array, external_forces)

@njit(cache=True)
def compute_friction_force_wall_array(position_collection: npt.NDArray[np.float64],
                                      velocity_collection: npt.NDArray[np.float64],
                                      nodal_total_force_collection: npt.NDArray[np.float64],
                                      normal_force_ndarray: npt.NDArray[np.float64],
                                      kinetic_mu_wall_array: npt.NDArray[np.float64],
                                      static_mu_wall_array: npt.NDArray[np.float64],
                                      velocity_threshold: float):
    """
    Optimized array version of wall friction force calculation with improved parallel processing.
    """
    N = position_collection.shape[1]
    friction_forces = np.zeros((3, N))
    
    n_mags = np.zeros(N)
    v_parallels = np.zeros(N)
    t_units = np.zeros((2, N))
    
    # First pass: compute magnitudes and unit vectors
    for i in prange(N):
        n_mags[i] = np.linalg.norm(normal_force_ndarray[:2, i])
        if n_mags[i] >= 1e-6:
            n_unit = normal_force_ndarray[:2, i] / n_mags[i]
            t_units[0, i] = -n_unit[1]
            t_units[1, i] = n_unit[0]
            v_parallels[i] = velocity_collection[0, i] * t_units[0, i] + velocity_collection[1, i] * t_units[1, i]
    
    # Second pass: compute friction forces
    for i in prange(N):
        if n_mags[i] < 1e-6:
            continue

        kinetic_coeff = kinetic_mu_wall_array[i]
        static_coeff = static_mu_wall_array[i]
            
        if np.abs(v_parallels[i]) > velocity_threshold:
            # Kinetic friction
            friction_forces[0, i] = -kinetic_coeff * n_mags[i] * np.sign(v_parallels[i]) * t_units[0, i]
            friction_forces[1, i] = -kinetic_coeff * n_mags[i] * np.sign(v_parallels[i]) * t_units[1, i]
        else:
            # Static friction
            F_static_max = static_coeff * n_mags[i]
            F_parallel = nodal_total_force_collection[0, i] * t_units[0, i] + nodal_total_force_collection[1, i] * t_units[1, i]
            friction_magnitude = min(np.abs(F_parallel), F_static_max)
            friction_forces[0, i] = -np.sign(F_parallel) * friction_magnitude * t_units[0, i]
            friction_forces[1, i] = -np.sign(F_parallel) * friction_magnitude * t_units[1, i]
    
    return friction_forces

@njit(cache=True)
def compute_friction_force_substrate_array(position_collection: npt.NDArray[np.float64],
                                           velocity_collection: npt.NDArray[np.float64],
                                           nodal_total_force_collection: npt.NDArray[np.float64],
                                           point_gravitational_force: float,
                                           kinetic_mu_substrate_array: npt.NDArray[np.float64],
                                           static_mu_substrate_array: npt.NDArray[np.float64],
                                           velocity_threshold: float):
    """
    Optimized array version of substrate friction force calculation with improved parallel processing.

    Returns:
        friction_forces: Shape (3, n_nodes)
    """
    N = position_collection.shape[1]
    friction_forces = np.zeros((3, N))
    
    v_mags = np.zeros(N)
    nodal_mags = np.zeros(N)
    v_units = np.zeros((2, N))
    
    # First pass: compute magnitudes and unit vectors
    for i in prange(N):
        v_mags[i] = np.linalg.norm(velocity_collection[:2, i])
        nodal_mags[i] = np.linalg.norm(nodal_total_force_collection[:2, i])
        if v_mags[i] >= velocity_threshold:
            v_units[0, i] = velocity_collection[0, i] / v_mags[i]
            v_units[1, i] = velocity_collection[1, i] / v_mags[i]
    
    # Second pass: compute friction forces
    for i in prange(N):
        kinetic_coeff: float = kinetic_mu_substrate_array[i]
        static_coeff: float = static_mu_substrate_array[i]
        
        if v_mags[i] >= velocity_threshold:
            # Kinetic friction
            friction_forces[0, i] = -kinetic_coeff * point_gravitational_force * v_units[0, i]
            friction_forces[1, i] = -kinetic_coeff * point_gravitational_force * v_units[1, i]
        else:
            # Static friction
            if nodal_mags[i] < 1e-6: # If the sum of internal and external forces are too small.
                continue
            F_static_max = static_coeff * point_gravitational_force
            if nodal_mags[i] < F_static_max:
                friction_forces[0, i] = -nodal_total_force_collection[0, i]
                friction_forces[1, i] = -nodal_total_force_collection[1, i]
            else:
                scale = F_static_max / nodal_mags[i] # Ensure maximum value Coulomb static friction enforced
                friction_forces[0, i] = -scale * nodal_total_force_collection[0, i]
                friction_forces[1, i] = -scale * nodal_total_force_collection[1, i]
    
    return friction_forces

@njit(cache=True)
def _batch_vector_sum(vector1: npt.NDArray[np.float64], vector2: npt.NDArray[np.float64]):
    """
    Optimized vector sum calculation.
    """
    blocksize = vector1.shape[1]
    output_vector = np.empty((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            output_vector[i, k] = vector1[i, k] + vector2[i, k]

    return output_vector

@njit(cache=True)
def update_nodal_force(boundary_response_force: npt.NDArray[np.float64], nodal_external_force: npt.NDArray[np.float64]):
    """
    Optimized nodal force update.

    Args:
        boundary_response_force: Shape (3, n_nodes)
        nodal_external_forces: Shape (3, n_nodes)

    Returns:
        nodal_external_forces: nodal_external_forces += boundary_response_force.
        Shape (3, n_nodes).
        It's an inplace addition due to njit's optimization, so the return nodal_external_force is not needed.
    """
    for i in range(3):
        for k in range(boundary_response_force.shape[1]):
            nodal_external_force[i, k] += boundary_response_force[i, k]

@njit(cache=True)
def compute_normal_force_array_fast(position_collection: npt.NDArray[np.float64],
                                    boundary_x: npt.NDArray[np.float64],
                                    boundary_y: npt.NDArray[np.float64],
                                    dist_threshold: float,
                                    boundary_stiffness: float,
                                    nodal_total_force_collection: npt.NDArray[np.float64],
                                    velocity_collection: npt.NDArray[np.float64],
                                    contact_damping: float):
    """
    Optimized version of normal force array calculation.
    """
    px = position_collection[0, :]  # Shape (n_nodes,) containing all nodes' x position
    py = position_collection[1, :]  # Shape (n_nodes,) containing all nodes' y position

    N = px.shape[0]
    M = boundary_x.shape[0]
    out = np.zeros((3, N)) # (xyz, n_nodes)

    for i in prange(N):
        x, y = px[i], py[i]
        
        # Point-in-polygon check
        inside = False
        j = M - 1 # The end index.

        # for k in n_boundary_points
        for k in range(M):
            yi, yj = boundary_y[k], boundary_y[j]
            xi, xj = boundary_x[k], boundary_x[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = k

        # Find closest point on boundary
        best_d2 = 1e18
        cx = 0.0
        cy = 0.0
        
        # Check one x, y point against all boundary points to see which boundary point is the closest
        for k in range(M-1):
            x1, y1 = boundary_x[k], boundary_y[k]
            x2, y2 = boundary_x[k+1], boundary_y[k+1]
            
            # Calculate vectors for projection
            vx = x2 - x1 # Boundary vector along each boundary point pair
            vy = y2 - y1
            wx = x - x1 # Boundary-to-robot vector
            wy = y - y1
            
            # Project point onto segment
            denom = vx*vx + vy*vy
            if denom > 0.0:
                t = (vx*wx + vy*wy) / denom
                t = max(0.0, min(1.0, t)) # Normalization via min max clipping
            else:
                t = 0.0
                
            # Calculate closest point on segment
            px0 = x1 + t*vx
            py0 = y1 + t*vy
            
            # Calculate squared distance to point
            dx = px0 - x
            dy = py0 - y
            d2 = dx*dx + dy*dy
            
            if d2 < best_d2:
                best_d2 = d2
                cx = px0
                cy = py0
        
        d = np.sqrt(best_d2)
        fx_n = 0.0
        fy_n = 0.0

        if not inside:
            # Point is outside channel - apply strong repulsive force
            dx, dy = cx - x, cy - y
            if d > 1e-6:
                ux, uy = dx/d, dy/d
                fx_n = boundary_stiffness * d * ux
                fy_n = boundary_stiffness * d * uy
        else:
            # Point is inside channel - apply normal and damping forces
            if d < dist_threshold:
                dx, dy = x - cx, y - cy
                nn = np.hypot(dx, dy)
                if nn > 1e-6:
                    ux, uy = dx/nn, dy/nn

                    # Calculate reaction force based on current nodal forces
                    ntx, nty = nodal_total_force_collection[0, i], nodal_total_force_collection[1, i]
                    reac = ntx*ux + nty*uy
                    frx = -reac*ux if reac < 0.0 else 0.0
                    fry = -reac*uy if reac < 0.0 else 0.0

                    # Combine penetration and damping forces
                    vx, vy = velocity_collection[0, i], velocity_collection[1, i]
                    vn = vx*ux + vy*uy
                    scale = (dist_threshold - d) * boundary_stiffness - contact_damping * abs(vn)
                    fx_n = frx + scale * ux
                    fy_n = fry + scale * uy

        out[0, i] = fx_n
        out[1, i] = fy_n

    return out