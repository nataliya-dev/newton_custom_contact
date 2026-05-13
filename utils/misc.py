import warp as wp

def get_object_on_slope_initial_pose(slope, up_slope_offset, object_bottom_offset_z):
    '''
    object_bottom_offset_z:
    Distance from the object's local origin to its lowest point,
    measured along the object's local +Z axis.
    '''
    local_z = wp.vec3(0.0, 0.0, 1.0)
    n = wp.quat_rotate(slope.rot0, local_z)
    base_center = slope.pos0 + n * (slope.hz + object_bottom_offset_z)
    local_y = wp.vec3(0.0, 1.0, 0.0)
    tangent = wp.quat_rotate(slope.rot0, local_y)
    center = base_center + tangent * up_slope_offset
    return center, slope.rot0
