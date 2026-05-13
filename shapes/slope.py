from newton._src.sim import builder
import warp as wp
import newton
from shapes.base_shape import ShapeBase

class Slope(ShapeBase):
    def __init__(self, pos0:wp.vec3, rot0:wp.quat, hx:float, hy:float, hz:float):
        self.pos0 = pos0 
        self.rot0 = rot0
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.particle_start_idx = None
        self.particle_end_idx = None
        self.particle_group_id = None

    def add_body(self, builder):
        cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
            is_solid=True
        )
        builder.add_shape_plane(
            -1,
            wp.transform(
                self.pos0,
                self.rot0,
            ),
            cfg=cfg,
            width=self.hx * 2,
            length=self.hy * 2,
        )
        
        return builder
    
    def add_mesh(self, builder):
        raise NotImplementedError
    
    def add_spheres(self, builder, num_spheres_x):
        """
        Add the slope as a grid of spheres to the builder.

        Args:
            builder: The ModelBuilder instance to add the spheres to.
            num_spheres: Number of spheres along the minimum dimension of the slope.
        Returns:
            Updated ModelBuilder with slope spheres added.
        """
        radius_mean = self.hx / num_spheres_x

        dim_x = num_spheres_x
        dim_y = int((2 * self.hy) / (2 * radius_mean)) + 1
        dim_z = 1

        cell_x = (2 * self.hx - 2 * radius_mean) / (dim_x - 1)
        cell_y = (2 * self.hy - 2 * radius_mean) / (dim_y - 1)
        cell_z = 0.0

        # Static: mass=0 for fixed particles
        mass_per_particle = 0.0

        pos_corner = self.pos0 + wp.quat_rotate(
            self.rot0, wp.vec3(-self.hx + radius_mean, -self.hy + radius_mean, self.hz - radius_mean)
        )
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
            radius_std=0.0
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

        return builder

    def add_morphit_spheres(self, builder, json_adrs):
        # For slope, we use mass=0 to make it static (unlike box which uses total_mass)
        self.particle_start_idx = len(builder.particle_q)
        
        # add_particle_volume will handle particle group registration
        builder.add_particle_volume(
            volume_data=json_adrs,
            pos=self.pos0,
            rot=self.rot0,
            vel=wp.vec3(0.0),
            total_mass=0.0  # Static slope
        )
        
        self.particle_end_idx = len(builder.particle_q)
        
        # Get the group ID that was assigned by add_particle_volume
        # All particles added by add_particle_volume will have the same group ID
        if self.particle_start_idx < self.particle_end_idx:
            self.particle_group_id = builder.particle_group[self.particle_start_idx]
        else:
            self.particle_group_id = None
        
        return builder
