import numpy as np
import warp as wp
import newton
from shapes.base_shape import ShapeBase


class Rod(ShapeBase):
    """
    A rod shape represented as a cylinder.
    
    Args:
        pos0: Initial position (center)
        rot0: Initial rotation
        radius: Cylinder radius
        length: Cylinder length (along z-axis by default)
        mass: Total mass of the rod
    """
    def __init__(self, pos0, rot0, radius, length, mass):
        self.pos0 = pos0
        self.rot0 = rot0
        self.radius = radius
        self.length = length
        self.particle_start_idx = None
        self.particle_end_idx = None
        self.particle_group_id = None

        # Compute inertia for cylinder
        density = mass / (np.pi * radius**2 * length)
        mass, com, I_m = newton.geometry.compute_inertia_shape(
            type=newton.GeoType.CYLINDER,
            # Functiona assumes half height of cylinder
            scale=(radius, length / 2.0),
            src=None,
            density=density,
            thickness=0.001,
            is_solid=True,
        )

        self.mass = mass
        self.I_m = I_m
        self.com = wp.transform_point(
            wp.transform(p=self.pos0, q=self.rot0), com)

    def add_body(self, builder):
        """Add rod as a rigid body"""
        density = self.mass / (np.pi * self.radius**2 * self.length)
        cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
            is_solid=True,
            density=density,
        )
        self.body_idx = builder.add_body(
            xform=wp.transform(p=self.pos0, q=self.rot0),
        )
        self.shape_idx = builder.add_shape_cylinder(
            self.body_idx,
            radius=self.radius,
            half_height=self.length / 2.0,
            cfg=cfg,
        )
        return builder

    def add_spheres(self, builder, num_spheres):
        """Add rod as a collection of spheres arranged along its length.
        
        Args:
            builder: ModelBuilder instance
            num_spheres: Number of spheres along the diameter of the rod.
                         Sphere radius will be: rod_radius / num_spheres
                         The outer spheres will be tangent to the rod's inner surface.
        """
        # Calculate sphere radius from num_spheres along diameter
        sphere_radius = self.radius / num_spheres
        sphere_diameter = 2.0 * sphere_radius
        
        # Along the length: fit as many spheres as possible
        num_along_length = max(1, int(self.length / sphere_diameter))
        
        # Generate radial positions so outer spheres are tangent to cylinder
        # For num_spheres along diameter, position them so:
        # - First sphere center at: -radius + sphere_radius (left edge at -radius)
        # - Last sphere center at: +radius - sphere_radius (right edge at +radius)
        # - Equal spacing of sphere_diameter between centers
        # Also generate y positions symmetrically to fill the cross-section
        offset = -self.radius + sphere_radius
        x_positions = [offset + k * sphere_diameter for k in range(num_spheres)]
        
        # Generate symmetric y positions with same spacing, filtered to fit in cylinder
        # As an example, if num_spheres=3 we should have the following:
        # |     O
        # |-- O O O ---- y=0
        # |     O
        y_positions = [0.0]
        
        for x in x_positions:
            k = 1
            # Check each x position for all the y values we can fit
            while True:
                y_candidate = k * sphere_diameter
                if np.sqrt(x**2 + y_candidate**2) + sphere_radius <= self.radius:
                    y_positions.append(y_candidate)
                    # Add symmetric negative position
                    y_positions.insert(0, -y_candidate)
                    k += 1
                else:
                    break

        
        z_positions = [(-self.length / 2.0 + sphere_radius) + k * sphere_diameter for k in range(num_along_length)]

        # Create grid mesh
        X, Y, Z = np.meshgrid(x_positions, y_positions, z_positions, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Filter points to only keep spheres that fit within cylinder bounds
        # A sphere of radius r centered at (x, y) fits in a cylinder of radius R if:
        # sqrt(x^2 + y^2) + r <= R
        valid_points = []
        for pt in points:
            x, y = pt[0], pt[1]
            dist_from_axis = np.sqrt(x**2 + y**2)
            if dist_from_axis + sphere_radius <= self.radius + 1e-6:  # Small tolerance for numerical errors
                valid_points.append(pt)
        
        if not valid_points:
            raise ValueError("No spheres fit within the rod dimensions with the given num_spheres")
        
        valid_points = np.array(valid_points)
        
        # Apply rotation and position offset to all points
        rot_mat = np.array(wp.quat_to_matrix(self.rot0)).reshape(3, 3)
        valid_points = valid_points @ rot_mat.T + np.array(self.pos0)
        
        # Create volume dict for add_particle_volume
        centers = valid_points.tolist()
        radii = [float(sphere_radius)] * len(centers)
        volume_dict = {"centers": centers, "radii": radii}
        
        # Use add_particle_volume to add particles
        self.particle_start_idx = len(builder.particle_q)
        
        builder.add_particle_volume(
            volume_data=volume_dict,
            pos=wp.vec3(0.0, 0.0, 0.0),  # Already transformed above
            rot=wp.quat_identity(float),  # Already transformed above
            vel=wp.vec3(0.0),
            total_mass=self.mass
        )
        
        self.particle_end_idx = len(builder.particle_q)
        
        # Retrieve the group_id from add_particle_volume's return value
        # and set up particle group tracking
        self.particle_group_id = builder.particle_group_count - 1
        
        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = np.sum(builder.particle_mass[self.particle_start_idx:self.particle_end_idx])
        return builder

    def add_morphit_spheres(self, builder, json_adrs):
        """Add rod using morphit JSON volume representation"""
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
