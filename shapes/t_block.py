import numpy as np
import warp as wp
import newton
from shapes.base_shape import ShapeBase


class TBlock(ShapeBase):
    """
    A T-shaped block consisting of a vertical stem and horizontal crossbar.
    
    The T-block is composed of two boxes:
    - Stem: vertical box (height > width, depth)
    - Crossbar: horizontal box extending from top of stem (width > height, depth)
    
    Args:
        pos0: Initial position (center of entire T-block)
        rot0: Initial rotation
        stem_hx: Half-width of stem
        stem_hy: Half-depth of stem
        stem_hz: Half-height of stem
        crossbar_hx: Half-width of crossbar (extends from stem)
        crossbar_hy: Half-depth of crossbar (same as stem depth)
        crossbar_hz: Half-height of crossbar
        mass: Total mass of the T-block
    """
    def __init__(self, pos0, rot0, stem_hx, stem_hy, stem_hz, 
                 crossbar_hx, crossbar_hy, crossbar_hz, mass):
        self.pos0 = pos0
        self.rot0 = rot0
        self.stem_hx = stem_hx
        self.stem_hy = stem_hy
        self.stem_hz = stem_hz
        self.crossbar_hx = crossbar_hx
        self.crossbar_hy = crossbar_hy
        self.crossbar_hz = crossbar_hz
        self.particle_start_idx = None
        self.particle_end_idx = None
        self.particle_group_id = None

        # Calculate volumes and mass distribution
        stem_volume = 8.0 * stem_hx * stem_hy * stem_hz
        crossbar_volume = 8.0 * crossbar_hx * crossbar_hy * crossbar_hz
        total_volume = stem_volume + crossbar_volume
        
        # Distribute mass proportionally to volume
        self.stem_mass = mass * (stem_volume / total_volume)
        self.crossbar_mass = mass * (crossbar_volume / total_volume)
        self.mass = mass

        # Compute center of mass of the compound shape (relative to pos0)
        # Stem COM is at origin (0, 0, 0)
        # Crossbar COM is at (0, 0, stem_hz + crossbar_hz)
        crossbar_offset_z = stem_hz + crossbar_hz
        com_z = (self.stem_mass * 0.0 + self.crossbar_mass * crossbar_offset_z) / self.mass
        com_offset = wp.vec3(0.0, 0.0, com_z)

        # Compute inertia for each component about its own center of mass
        stem_density = self.stem_mass / stem_volume
        _, _, I_stem = newton.geometry.compute_inertia_shape(
            type=newton.GeoType.BOX,
            scale=(stem_hx, stem_hy, stem_hz),
            src=None,
            density=stem_density,
            thickness=0.001,
            is_solid=True,
        )

        crossbar_density = self.crossbar_mass / crossbar_volume
        _, _, I_crossbar = newton.geometry.compute_inertia_shape(
            type=newton.GeoType.BOX,
            scale=(crossbar_hx, crossbar_hy, crossbar_hz),
            src=None,
            density=crossbar_density,
            thickness=0.001,
            is_solid=True,
        )

        # Apply parallel axis theorem to shift inertias to compound COM
        # Distance from stem COM to compound COM
        d_stem = wp.vec3(0.0, 0.0, 0.0) - com_offset
        # Distance from crossbar COM to compound COM
        d_crossbar = wp.vec3(0.0, 0.0, crossbar_offset_z) - com_offset

        # Parallel axis theorem: I_shifted = I_cm + m * (||d||^2 * I - d ⊗ d)
        # For simplicity, compute diagonal elements
        def apply_parallel_axis_to_diag(I, m, d):
            """Apply parallel axis theorem to inertia tensor"""
            I_parallel = I.copy() if isinstance(I, np.ndarray) else np.array(I)
            d_sq = np.dot(d, d)
            I_parallel[0] += m * (d[1]**2 + d[2]**2)
            I_parallel[1] += m * (d[0]**2 + d[2]**2)
            I_parallel[2] += m * (d[0]**2 + d[1]**2)
            return I_parallel

        # Assume I_stem and I_crossbar are diagonal (or can be treated as such)
        try:
            I_stem_diag = np.array([I_stem[0, 0], I_stem[1, 1], I_stem[2, 2]]) if isinstance(I_stem, np.ndarray) and I_stem.ndim == 2 else np.array(I_stem)
            I_crossbar_diag = np.array([I_crossbar[0, 0], I_crossbar[1, 1], I_crossbar[2, 2]]) if isinstance(I_crossbar, np.ndarray) and I_crossbar.ndim == 2 else np.array(I_crossbar)
        except:
            I_stem_diag = np.array(I_stem) if isinstance(I_stem, (list, tuple)) else np.diag(I_stem) if isinstance(I_stem, np.ndarray) else I_stem
            I_crossbar_diag = np.array(I_crossbar) if isinstance(I_crossbar, (list, tuple)) else np.diag(I_crossbar) if isinstance(I_crossbar, np.ndarray) else I_crossbar

        I_stem_shifted = apply_parallel_axis_to_diag(I_stem_diag, self.stem_mass, d_stem)
        I_crossbar_shifted = apply_parallel_axis_to_diag(I_crossbar_diag, self.crossbar_mass, d_crossbar)

        # Sum inertias
        self.I_m = I_stem_shifted + I_crossbar_shifted
        self.com = wp.transform_point(
            wp.transform(p=self.pos0, q=self.rot0), com_offset)

    def add_body(self, builder):
        """Add T-block as a single rigid body with stem and crossbar shapes"""
        # Compute center of mass
        crossbar_offset_z = self.stem_hz + self.crossbar_hz
        com_z = (self.stem_mass * 0.0 + self.crossbar_mass * crossbar_offset_z) / self.mass
        
        # Create a single body at the center of mass
        body_pos = self.pos0 + wp.quat_rotate(self.rot0, wp.vec3(0.0, 0.0, com_z))
        self.body_idx = builder.add_body(
            xform=wp.transform(p=body_pos, q=self.rot0),
        )

        # Add stem shape (positioned below COM in body frame)
        stem_offset_z = -com_z
        stem_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
            is_solid=True,
            density=self.stem_mass / (8.0 * self.stem_hx * self.stem_hy * self.stem_hz),
        )
        builder.add_shape_box(
            self.body_idx,
            hx=self.stem_hx,
            hy=self.stem_hy,
            hz=self.stem_hz,
            cfg=stem_cfg,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, stem_offset_z), q=wp.quat_identity()),
        )

        # Add crossbar shape (positioned above COM in body frame)
        crossbar_offset_z_rel = crossbar_offset_z - com_z
        crossbar_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
            is_solid=True,
            density=self.crossbar_mass / (8.0 * self.crossbar_hx * self.crossbar_hy * self.crossbar_hz),
        )
        builder.add_shape_box(
            self.body_idx,
            hx=self.crossbar_hx,
            hy=self.crossbar_hy,
            hz=self.crossbar_hz,
            cfg=crossbar_cfg,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, crossbar_offset_z_rel), q=wp.quat_identity()),
        )
        
        return builder

    def add_spheres(self, builder, num_spheres):
        """Add T-block as a collection of uniformly-sized spheres (stem + crossbar)"""
        # Use a consistent sphere radius for both stem and crossbar
        # based on the smaller dimension to ensure uniform coverage
        min_dimension = min(self.stem_hx, self.stem_hy, self.stem_hz, 
                           self.crossbar_hx, self.crossbar_hy, self.crossbar_hz)
        uniform_radius = min_dimension / num_spheres
        
        # Calculate how many particles fit in each section
        stem_volume = 8.0 * self.stem_hx * self.stem_hy * self.stem_hz
        crossbar_volume = 8.0 * self.crossbar_hx * self.crossbar_hy * self.crossbar_hz
        total_volume = stem_volume + crossbar_volume
        
        # Distribute total particles proportionally by volume
        total_particles = num_spheres ** 3
        stem_particles = max(1, int(total_particles * stem_volume / total_volume))
        crossbar_particles = max(1, total_particles - stem_particles)
        
        self.particle_start_idx = len(builder.particle_q)

        # Add stem particles - fit particles uniformly
        # Offset corner inward by one radius so sphere surfaces align with box surfaces
        stem_dim_x = max(1, int(np.ceil(2.0 * self.stem_hx / (2.0 * uniform_radius))))
        stem_dim_y = max(1, int(np.ceil(2.0 * self.stem_hy / (2.0 * uniform_radius))))
        stem_dim_z = max(1, int(np.ceil(2.0 * self.stem_hz / (2.0 * uniform_radius))))
        
        if stem_dim_x > 1:
            stem_cell_x = (2.0 * self.stem_hx - 2.0 * uniform_radius) / (stem_dim_x - 1)
        else:
            stem_cell_x = 0.0
        if stem_dim_y > 1:
            stem_cell_y = (2.0 * self.stem_hy - 2.0 * uniform_radius) / (stem_dim_y - 1)
        else:
            stem_cell_y = 0.0
        if stem_dim_z > 1:
            stem_cell_z = (2.0 * self.stem_hz - 2.0 * uniform_radius) / (stem_dim_z - 1)
        else:
            stem_cell_z = 0.0
        
        stem_mass_per_particle = self.stem_mass / (stem_dim_x * stem_dim_y * stem_dim_z)
        stem_pos_corner = self.pos0 + wp.quat_rotate(self.rot0, wp.vec3(
            -self.stem_hx + uniform_radius,
            -self.stem_hy + uniform_radius,
            -self.stem_hz + uniform_radius
        ))
        
        builder.add_particle_grid(
            pos=wp.vec3(stem_pos_corner[0], stem_pos_corner[1], stem_pos_corner[2]),
            rot=self.rot0,
            vel=wp.vec3(0.0),
            dim_x=stem_dim_x,
            dim_y=stem_dim_y,
            dim_z=stem_dim_z,
            cell_x=stem_cell_x,
            cell_y=stem_cell_y,
            cell_z=stem_cell_z,
            mass=stem_mass_per_particle,
            jitter=0.0,
            radius_mean=uniform_radius,
            radius_std=0.0,
        )

        # Add crossbar particles - fit particles uniformly
        # Offset corner inward by one radius so sphere surfaces align with box surfaces
        crossbar_dim_x = max(1, int(np.ceil(2.0 * self.crossbar_hx / (2.0 * uniform_radius))))
        crossbar_dim_y = max(1, int(np.ceil(2.0 * self.crossbar_hy / (2.0 * uniform_radius))))
        crossbar_dim_z = max(1, int(np.ceil(2.0 * self.crossbar_hz / (2.0 * uniform_radius))))
        
        if crossbar_dim_x > 1:
            crossbar_cell_x = (2.0 * self.crossbar_hx - 2.0 * uniform_radius) / (crossbar_dim_x - 1)
        else:
            crossbar_cell_x = 0.0
        if crossbar_dim_y > 1:
            crossbar_cell_y = (2.0 * self.crossbar_hy - 2.0 * uniform_radius) / (crossbar_dim_y - 1)
        else:
            crossbar_cell_y = 0.0
        if crossbar_dim_z > 1:
            crossbar_cell_z = (2.0 * self.crossbar_hz - 2.0 * uniform_radius) / (crossbar_dim_z - 1)
        else:
            crossbar_cell_z = 0.0
        
        crossbar_mass_per_particle = self.crossbar_mass / (crossbar_dim_x * crossbar_dim_y * crossbar_dim_z)
        crossbar_offset_z = self.stem_hz + self.crossbar_hz
        crossbar_pos_corner = self.pos0 + wp.quat_rotate(self.rot0, wp.vec3(
            -self.crossbar_hx + uniform_radius,
            -self.crossbar_hy + uniform_radius,
            crossbar_offset_z - self.crossbar_hz + uniform_radius
        ))

        builder.add_particle_grid(
            pos=wp.vec3(crossbar_pos_corner[0], crossbar_pos_corner[1], crossbar_pos_corner[2]),
            rot=self.rot0,
            vel=wp.vec3(0.0),
            dim_x=crossbar_dim_x,
            dim_y=crossbar_dim_y,
            dim_z=crossbar_dim_z,
            cell_x=crossbar_cell_x,
            cell_y=crossbar_cell_y,
            cell_z=crossbar_cell_z,
            mass=crossbar_mass_per_particle,
            jitter=0.0,
            radius_mean=uniform_radius,
            radius_std=0.0,
        )

        self.particle_end_idx = len(builder.particle_q)

        # Register as new particle group
        group_id = builder.particle_group_count
        builder.particle_group_count += 1
        
        for i in range(self.particle_start_idx, self.particle_end_idx):
            builder.particle_group[i] = group_id

        builder.particle_groups[group_id] = list(range(self.particle_start_idx, self.particle_end_idx))
        self.particle_group_id = group_id

        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = (self.stem_mass + self.crossbar_mass)
        return builder

    def add_morphit_spheres(self, builder, json_adrs):
        """Add T-block using morphit JSON volume representation"""
        self.particle_start_idx = len(builder.particle_q)

        builder.add_particle_volume(
            volume_data=json_adrs,
            pos=self.pos0,
            rot=self.rot0,
            vel=wp.vec3(0.0),
            total_mass=self.mass
        )
        self.particle_end_idx = len(builder.particle_q)

        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = np.sum(builder.particle_mass)

        if self.particle_start_idx < self.particle_end_idx:
            self.particle_group_id = builder.particle_group[self.particle_start_idx]
        else:
            self.particle_group_id = None
        return builder

    def add_mesh(self, builder):
        raise NotImplementedError
