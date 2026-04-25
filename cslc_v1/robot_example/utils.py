import numpy as np

from newton.solvers import SolverSemiImplicit
from newton.solvers import SolverMuJoCo
import newton
from newton._src.geometry.types import GeoType

from cslc_v1.robot_example.config import SceneParams



_SEP = "─" * 60


def _log(msg, indent=0):
    print(f"  {'  ' * indent}│ {msg}")


def _section(title):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")
    
    
CSLC_FLAG = 1 << 5


def find_body_in_builder(builder, name):
    return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))


def inspect_model(model, label=""):
    """Print body/shape/joint summary for a finalized model.

    Each shape line shows its geometry type, the body it's attached to, and
    a ``[CSLC]`` marker if the shape carries the CSLC contact flag — useful
    for confirming pad shapes survived ``add_builder`` and got the right
    contact model attached.
    """
    _log(
        f"Model '{label}': {model.body_count} bodies, {model.shape_count} shapes, "
        f"{model.joint_count} joints, {model.joint_dof_count} DOFs"
    )
    st = model.shape_type.numpy()
    sf = model.shape_flags.numpy()
    sb = model.shape_body.numpy()
    for i in range(model.shape_count):
        cslc = " [CSLC]" if sf[i] & CSLC_FLAG else ""
        geo_name = GeoType(int(st[i])).name
        _log(f"  shape {i}: {geo_name}  body={sb[i]}{cslc}", 1)


def count_active_contacts(contacts):
    n = int(contacts.rigid_contact_count.numpy()[0])
    return int(np.sum(contacts.rigid_contact_shape0.numpy()[:n] >= 0)) if n else 0


def recalibrate_cslc_kc_per_pad(model, contact_fraction):
    """Override per-sphere kc on a per-pad basis (mirror of the squeeze_test
    helper).  Sets each pad's aggregate stiffness at uniform contact equal to
    `ke_bulk`, using the empirical contact fraction for this scene."""
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None

    d = handler.cslc_data
    # Read ke from the FIRST CSLC-flagged shape, not shape 0 (which may be a
    # ground plane in lift_test).  Squeeze_test happens to have shape 0 == a
    # CSLC pad, so its helper got away with `shape_material_ke[0]` directly.
    shape_flags = model.shape_flags.numpy()
    cslc_shape_idx = next(
        (i for i in range(model.shape_count) if (shape_flags[i] & CSLC_FLAG)),
        0,
    )
    ke_bulk = float(model.shape_material_ke.numpy()[cslc_shape_idx])
    shape_ids = d.sphere_shape.numpy()
    is_surface = d.is_surface.numpy()
    n_pads = int(len(np.unique(shape_ids)))
    n_surface_per_pad = int(is_surface.sum()) // max(n_pads, 1)
    n_contact_per_pad = max(int(n_surface_per_pad * contact_fraction), 1)

    ka = float(d.ka)
    denom = n_contact_per_pad * ka - ke_bulk
    if denom <= 0.0:
        new_kc = ke_bulk / max(n_contact_per_pad, 1)
        derivation = "fallback (denom<=0): kc = ke/N"
    else:
        new_kc = ke_bulk * ka / denom
        derivation = "exact: kc = ke*ka/(N*ka - ke)"

    old_kc = float(d.kc)
    d.kc = new_kc
    keff = new_kc * ka / (ka + new_kc)
    aggregate_per_pad = n_contact_per_pad * keff
    _log(f"CSLC RECAL: pads={n_pads}  N_contact_per_pad={n_contact_per_pad}  "
         f"({derivation})")
    _log(f"            kc: {old_kc:.1f}  →  {new_kc:.1f} N/m  "
         f"keff={keff:.1f}  agg/pad={aggregate_per_pad:.0f} (target={ke_bulk:.0f})")
    return new_kc


def read_cslc_state(model):
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    is_surf = d.is_surface.numpy() == 1
    deltas = d.sphere_delta.numpy()[is_surf]
    pen = handler.raw_penetration.numpy()[is_surf]
    active = pen > 0
    return {
        "n_active": int(active.sum()), "n_surface": int(is_surf.sum()),
        "max_delta_mm": float(deltas.max()) * 1e3 if len(deltas) else 0,
        "max_pen_mm": float(pen.max()) * 1e3 if len(pen) else 0,
    }


class SimDiagnostics:
    def __init__(
        self,
        sphere_body_idx: int,
        pad_shape_idx: int,
        model,
        sphere_mass: float,
        gravity_z: float,
        print_every_frames: int = 6,
    ):
        self.sphere_body_idx = sphere_body_idx
        self.pad_body_idx = int(model.shape_body.numpy()[pad_shape_idx])
        self._pad_local_xform = model.shape_transform.numpy()[pad_shape_idx]
        self.sphere_mass = sphere_mass
        self.gravity_z = gravity_z
        self.print_every_frames = print_every_frames
        self._prev_sphere_z = None
        self._prev_pad_z = None
        self._prev_vz_sphere = None

        # Public, UI-readable snapshot of the latest log() call. Refreshed
        # every frame (even on non-print frames) so the viewer side panel
        # reflects the current sim state, not the last printed sample.
        self.frame_idx: int = 0
        self.phase_name: str = ""
        self.sphere_z: float = 0.0
        self.pad_z: float = 0.0
        self.delta_z: float = 0.0
        self.contact_count: int = 0
        self.cslc_state: dict | None = None

    @staticmethod
    def _shape_world_z(body_xform: np.ndarray, local_xform: np.ndarray) -> float:
        # body_xform / local_xform layout: (px, py, pz, qx, qy, qz, qw).
        # Compose: world = body_pos + rotate(body_quat, local_pos).
        body_pos = body_xform[:3]
        qx, qy, qz, qw = body_xform[3:7]
        local_pos = local_xform[:3]
        qv = np.array([qx, qy, qz])
        cross1 = np.cross(qv, local_pos)
        rotated = local_pos + 2.0 * qw * cross1 + 2.0 * np.cross(qv, cross1)
        return float((body_pos + rotated)[2])

    def log(
        self,
        frame_idx: int,
        phase_name: str,
        body_q_np: np.ndarray,
        active_contacts: int,
        frame_dt: float,
        cslc_state: dict | None = None,
    ):
        sphere_z = float(body_q_np[self.sphere_body_idx][2])
        pad_z = self._shape_world_z(body_q_np[self.pad_body_idx], self._pad_local_xform)
        delta_z = sphere_z - pad_z

        # Update the UI snapshot every frame so the side panel never lags
        # behind the actual sim state, even when we're not printing this frame.
        self.frame_idx = frame_idx
        self.phase_name = phase_name
        self.sphere_z = sphere_z
        self.pad_z = pad_z
        self.delta_z = delta_z
        self.contact_count = active_contacts
        self.cslc_state = cslc_state

        if frame_idx % self.print_every_frames != 0:
            return

        dt = self.print_every_frames * frame_dt
        if self._prev_sphere_z is None:
            vz_sphere = 0.0
            vz_pad = 0.0
        else:
            vz_sphere = (sphere_z - self._prev_sphere_z) / dt
            vz_pad = (pad_z - self._prev_pad_z) / dt

        if self._prev_vz_sphere is None:
            az_sphere = 0.0
        else:
            az_sphere = (vz_sphere - self._prev_vz_sphere) / dt

        self._prev_sphere_z = sphere_z
        self._prev_pad_z = pad_z
        self._prev_vz_sphere = vz_sphere

        g_mag = abs(self.gravity_z)
        f_contact_z = self.sphere_mass * (az_sphere + g_mag)
        weight_n = self.sphere_mass * g_mag

        cslc_str = (
            f"  cslc={cslc_state['n_active']}/{cslc_state['n_surface']}"
            if cslc_state
            else ""
        )

        _log(
            f"[{phase_name:15s}] step={frame_idx:5d}  "
            f"sz={sphere_z:+.4f} pz={pad_z:+.4f} Δz={delta_z * 1e3:+6.2f}mm  "
            f"vz_s={vz_sphere:+.4f} vz_p={vz_pad:+.4f} "
            f"az_s={az_sphere:+6.2f}m/s²  "
            f"F_c={f_contact_z:+6.2f}N (W={weight_n:.2f}N)  "
            f"n={active_contacts}{cslc_str}"
        )


def get_sphere_cfg_not_hyrdo(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density)
 

def get_sphere_cfg_hydro(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density,
        kh=p.kh, is_hydroelastic=True, sdf_max_resolution=p.sdf_resolution)
       

def point_pad_cfg(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density)
    return cfg


def get_cslc_pad_cfg(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        is_cslc=True,
        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    return cfg


def get_hydro_pad_cfg(p: SceneParams):
    pad_cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        kh=p.kh, is_hydroelastic=True, sdf_max_resolution=p.sdf_resolution)
    return pad_cfg


def make_solver(model, solver_name, scene_params: SceneParams):    
    if solver_name == "mujoco":
        ncon = 5000
        if model.shape_cslc_spacing is not None:
            spacing = model.shape_cslc_spacing.numpy()
            flags = model.shape_flags.numpy()
            scale = model.shape_scale.numpy()
            for i in range(model.shape_count):
                if not (flags[i] & CSLC_FLAG):
                    continue
                sp = float(spacing[i])
                if sp <= 0:
                    continue
                hx, hy, hz = (float(scale[i][j]) for j in range(3))
                nx, ny, nz = (max(int(round(2*h/sp))+1, 2)
                              for h in [hx, hy, hz])
                interior = max(nx-2, 0)*max(ny-2, 0)*max(nz-2, 0)
                ncon += nx*ny*nz - interior
        
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            solver="cg",
            integrator="implicitfast",
            cone="elliptic",
            iterations=20,
            ls_iterations=100,
            njmax=ncon,
            nconmax=ncon,
            impratio=1000.0
        )
        # From lift_test.py, but the sphere slips when CSLC is used.
        # return SolverMuJoCo(model, use_mujoco_contacts=False,
        #                     solver="cg", integrator="implicitfast",
        #                     cone="elliptic",
        #                     iterations=100, ls_iterations=10,
        #                     njmax=ncon, nconmax=ncon)
    elif solver_name == "semi":
        return SolverSemiImplicit(model)
    raise ValueError(f"Unknown solver: {solver_name}")
