import warp as wp
import math
import numpy as np
from shapes.base_shape import ShapeBase
from utils.mesh_helper import calculate_physics_properties_for_obj
from pxr import Usd
import newton.usd


class TorusBase(ShapeBase):
    def __init__(self, pos0, rot0, obj_path, usd_path):
        self.pos0 = pos0
        self.rot0 = rot0
        self.transform = wp.transform(
            p=self.pos0,
            q=self.rot0
        )
        self.obj_path = obj_path
        self.usd_path = usd_path
        self.mass, com, self.I_m = calculate_physics_properties_for_obj(self.obj_path)
        self.com = wp.transform_point(self.transform, com)

    def add_mesh(self, builder):
        usd_stage = Usd.Stage.Open(self.usd_path)
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/mesh"))
        body_mesh = builder.add_body(xform=wp.transform(p=self.pos0, q=self.rot0), key="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)
        return builder

    def add_spheres(self, builder, radius):
        builder.manual_sphere_packing(self.obj_path,
                                      radius=radius, spacing=radius*2,  # TODO: assumes no overlap
                                      total_mass=self.mass,
                                      pos=self.transform.p,
                                      rot=self.transform.q
                                      )
        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = np.sum(builder.particle_mass)
        return builder

    def add_morphit_spheres(self, builder, json_adrs):
        builder.add_particle_volume(
            volume_data=json_adrs,
            pos=self.pos0,
            rot = self.rot0,
            vel=wp.vec3(0.0),
            total_mass=self.mass
        )
        self.particle_q_init = builder.particle_q.copy()
        self.particle_com_init = np.mean(self.particle_q_init, axis=0)
        self.particle_mass_sum = np.sum(builder.particle_mass)
        return builder


class TorusB(TorusBase):
    obj_path = 'assets/hiro/torus/torus_bg.obj'
    usd_path = 'assets/hiro/torus/torus_bg.usd'
    
    def __init__(self, pos0, rot0):
        super().__init__(pos0, rot0, self.obj_path, self.usd_path)


class TorusS(TorusBase):
    obj_path = 'assets/hiro/torus/torus_sm.obj'
    usd_path = 'assets/hiro/torus/torus_sm.usd'
    
    def __init__(self, pos0, rot0):
        super().__init__(pos0, rot0, self.obj_path, self.usd_path)
