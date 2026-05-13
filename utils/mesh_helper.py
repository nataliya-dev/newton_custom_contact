import newton
import numpy as np
import warp as wp
import json

def _load_mesh_vertices_indices(obj_path):
    """Load a triangle mesh from a file. Uses trimesh (already in the env)."""
    import trimesh
    m = trimesh.load(obj_path, force="mesh")
    if m.is_empty:
        raise ValueError(f"Failed to load mesh from file: {obj_path}")
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int32).flatten()


def calculate_physics_properties_for_obj(obj_path):
    # Assumes a fixed density for all objects
    mesh_points, mesh_indices = _load_mesh_vertices_indices(obj_path)
    mesh = newton.Mesh(mesh_points, mesh_indices)
    shape_cfg = newton.ModelBuilder.ShapeConfig()

    mass, com, I_m = newton.geometry.compute_inertia_shape(
        type=newton.GeoType.MESH,
        scale=wp.vec3(1.0, 1.0, 1.0),
        src=mesh,
        density=shape_cfg.density,
        thickness=0.001,  # default thickness for hollow shapes; unused for solid
        is_solid=True,
    )
    return mass, com, I_m


def calculate_z_lowest_mesh(obj_path, transform=None):
    mesh_points, _ = _load_mesh_vertices_indices(obj_path)
    if transform is not None:
        R = np.array(wp.quat_to_matrix(transform.q)).reshape((3, 3))
        mesh_points = (R @ mesh_points.T).T
    z_offset_from_ground = -np.min(mesh_points[:, 2])
    return np.float32(z_offset_from_ground)

def calculate_z_lowest_sphere_set(json_adrs):
    with open(json_adrs, "r") as f:
        data = json.load(f)
    centers = np.array(data["centers"], dtype=np.float32)
    radii = np.array(data["radii"], dtype=np.float32)
    z_offset_from_ground = -np.min(centers[:, 2] - radii)
    return np.float32(z_offset_from_ground)

def calculate_box_dimensions_from_sphere_set(json_adrs):
    '''
    Calculates the half-extents of a box that can enclose the sphere set defined in the given JSON file.
    Must only be used for sphere sets that are roughly box-shaped.
    '''
    with open(json_adrs, "r") as f:
        data = json.load(f)
    centers = np.array(data["centers"], dtype=np.float32)
    radii = np.array(data["radii"], dtype=np.float32)

    x_min = np.min(centers[:, 0] - radii)
    x_max = np.max(centers[:, 0] + radii)
    y_min = np.min(centers[:, 1] - radii)
    y_max = np.max(centers[:, 1] + radii)
    z_min = np.min(centers[:, 2] - radii)
    z_max = np.max(centers[:, 2] + radii)

    hx = (x_max - x_min) / 2.0
    hy = (y_max - y_min) / 2.0
    hz = (z_max - z_min) / 2.0

    return np.float32(hx), np.float32(hy), np.float32(hz)