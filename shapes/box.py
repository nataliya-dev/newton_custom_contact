import numpy as np
import warp as wp
import newton
from shapes.base_shape import ShapeBase


class Box(ShapeBase):
    def __init__(self, pos0, rot0, hx, hy, hz, mass):
        self.pos0 = pos0
        self.rot0 = rot0
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.particle_start_idx = None
        self.particle_end_idx = None
        self.particle_group_id = None

        density = mass / (8.0 * hx * hy * hz)
        mass, com, I_m = newton.geometry.compute_inertia_shape(
            type=newton.GeoType.BOX,
            scale=(hx, hy, hz),
            src=None,
            density=density,
            thickness=0.001,  # default thickness for hollow shapes; unused for solid
            is_solid=True,
        )

        self.mass = mass
        self.I_m = I_m
        self.com = wp.transform_point(
            wp.transform(p=self.pos0, q=self.rot0), com)

    def add_body(self, builder):
        # density = mass / volume
        density = self.mass / (8.0 * self.hx * self.hy * self.hz)
        cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
            is_solid=True,
            density=density,
        )
        self.body_idx = builder.add_body(
            xform=wp.transform(p=self.pos0, q=self.rot0),
        )
        builder.add_shape_box(
            self.body_idx,
            hx=self.hx,
            hy=self.hy,
            hz=self.hz,
            cfg=cfg,
        )
        return builder

    def add_spheres(self, builder, num_spheres):
        dim_x = dim_y = dim_z = num_spheres
        radius_mean = self.hx / num_spheres
        if num_spheres == 1:
            cell_x = cell_y = cell_z = 0.0
        else:
            cell_x = (2 * self.hx - 2 * radius_mean) / (dim_x - 1)
            cell_y = (2 * self.hy - 2 * radius_mean) / (dim_y - 1)
            cell_z = (2 * self.hz - 2 * radius_mean) / (dim_z - 1)
        total_mass = self.mass
        num_particles = dim_x * dim_y * dim_z
        mass_per_particle = total_mass / num_particles
        pos_corner = self.pos0 + \
            wp.quat_rotate(self.rot0, wp.vec3(-self.hx + radius_mean, -
                           self.hy + radius_mean, -self.hz + radius_mean))

        self.particle_start_idx = len(builder.particle_q)

        builder.add_particle_grid(
            pos=wp.vec3(pos_corner[0], pos_corner[1], pos_corner[2]),
            rot=self.rot0,
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            mass=mass_per_particle,
            jitter=0.0,
            radius_mean=radius_mean,
            radius_std=0.0,
        )
        self.particle_end_idx = len(builder.particle_q)

        # Register as new particle group
        group_id = builder.particle_group_count
        builder.particle_group_count += 1
        # Update particle_group array for all particles in this range
        for i in range(self.particle_start_idx, self.particle_end_idx):
            builder.particle_group[i] = group_id
        # Store reverse mapping
        builder.particle_groups[group_id] = list(range(self.particle_start_idx, self.particle_end_idx))
        # Store group ID on this slope instance
        self.particle_group_id = group_id

        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = mass_per_particle * num_particles
        return builder

    def add_morphit_spheres(self, builder, json_adrs):
        self.particle_start_idx = len(builder.particle_q)

        builder.add_particle_volume(
            volume_data=json_adrs,
            pos = self.pos0,
            rot = self.rot0,
            vel=wp.vec3(0.0),
            total_mass=self.mass
        )
        self.particle_end_idx = len(builder.particle_q)

        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = np.sum(builder.particle_mass)

        # Get the group ID that was assigned by add_particle_volume
        # All particles added by add_particle_volume will have the same group ID
        if self.particle_start_idx < self.particle_end_idx:
            self.particle_group_id = builder.particle_group[self.particle_start_idx]
        else:
            self.particle_group_id = None
        return builder

    def add_mesh(self, builder):
        raise NotImplementedError
