from dataclasses import dataclass
import numpy as np

import newton

_SEP = "─" * 60


def _log(msg, indent=0):
    print(f"  {'  ' * indent}│ {msg}")


def _section(title):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")
    
    
CSLC_FLAG = 1 << 5


def find_body_in_builder(builder, name):
    return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))


def inspect_model(model, label=""):
    GEO = {0: "PLANE", 1: "MESH", 3: "SPHERE", 4: "CAPSULE", 7: "BOX"}
    _log(f"Model '{label}': {model.body_count} bodies, {model.shape_count} shapes, "
         f"{model.joint_count} joints, {model.joint_dof_count} DOFs")
    st, sf, sb = model.shape_type.numpy(
    ), model.shape_flags.numpy(), model.shape_body.numpy()
    for i in range(model.shape_count):
        cslc = " [CSLC]" if sf[i] & CSLC_FLAG else ""
        _log(f"  shape {i}: {GEO.get(int(st[i]), '?')}  body={sb[i]}{cslc}", 1)
        
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
    

@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # Sphere
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0
    
    
    penetration_depth = 0.02

    # # Table geometry and world position
    # table_center_x: float = 0.0
    # table_center_y: float = 0.0
    # table_surface_z: float = 0.3   # z of the top surface [m]
    # table_hx: float = 0.2
    # table_hy: float = 0.2
    # table_hz: float = 0.15

    # # Robot mount (position derived from table)
    # robot_standoff: float = 0.2    # gap from robot base to table front edge [m]
    # robot_mount_z: float = 0.0    # floor-mount height offset [m]

    # Pads (box half-extents and local finger-frame z-offset along finger axis)
    pad_hx: float = 0.01
    pad_hy: float = 0.002
    pad_hz: float = 0.01
    pad_local_z: float = 0.04525   # local z along finger axis — rubber-tip centre [m]
    pad_density: float = 1000.0


    reach_duration: float = 3.0

    approach_gap: float = 0.10
    approach_speed: float = 20e-3 / 1.5   # 13.33 mm/s → travels 20 mm over 1.5 s
    approach_duration: float = 1.5

    squeeze_speed: float = 1e-3 / 0.5     # 2 mm/s → travels 1 mm over 0.5 s
    squeeze_duration: float = 0.5

    lift_speed: float = 0.015
    lift_duration: float = 1.5

    # Smooth the target-velocity transition between SQUEEZE and LIFT over
    # this many seconds.  Without ramping, the target velocity jumps 0 →
    # lift_speed in a single timestep; the high-gain position PD drive
    # turns that into an ~impulsive pad velocity, which saturates friction
    # (μ·Fn) against the stationary sphere and launches it ballistically.
    # A 250 ms smoothstep puts the transition well above the drive's own
    # time constant (√(m/ke) ≈ 1 ms) so the pad follows the target faithfully
    # and the sphere accelerates gradually.
    lift_ramp_duration: float = 0.25

    hold_duration: float = 1.0

    ke: float = 5.0e4     # matched to squeeze_test
    kd: float = 5.0e2     # matched to squeeze_test
    kf: float = 100.0
    mu: float = 0.5

    # Joint drive stiffness
    drive_ke: float = 5.0e4
    drive_kd: float = 1.0e3

    # CSLC tuning — matched to squeeze_test.py
    cslc_spacing: float = 0.005
    cslc_ka: float = 5000.0
    cslc_kl: float = 500.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 20
    cslc_alpha: float = 0.6
    # Per-pad contact fraction for kc recalibration after handler build.
    # The lift scene uses face_pen ≈ 1 mm vs squeeze_test's 15 mm, so the
    # active patch contains only ~5 spheres per pad (vs 87 in squeeze).
    # The handler's default cf=0.3 over-estimates the active count by ~11×,
    # leaving CSLC's per-pad aggregate stiffness an order of magnitude below
    # ke_bulk.  Setting cf to the empirical fraction restores the calibration
    # invariant: per-pad aggregate stiffness = ke_bulk.
    # Set None to keep the handler's default kc.
    cslc_contact_fraction: float | None = 0.025

    # Hydroelastic modulus [Pa] for the hydro contact model.  See section 9
    # in cslc_v1/convo_april_19.md for the kh stability sweep — 1e8 is
    # silicone-rubber-stiff and stable; 1e10 ejects the sphere.
    kh: float = 1.0e8
    sdf_resolution: int = 64

    # Integration
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

    # ── Diagnostic / decoupling options ────────────────────────────────────
    # no_ground:     skip add_ground_plane().  Eliminates the ground-contact
    #                elastic-rebound confounder (previous agent's hypothesis C).
    # start_gripped: place the sphere in the air with pads already at
    #                squeeze-end position, touching it.  Skips APPROACH
    #                and SQUEEZE; test begins at LIFT t=0.
    # The two together isolate "can friction carry the sphere?" from "is
    # the launch triggered by ground release?".
    no_ground: bool = False
    start_gripped: bool = False
    # Warm-start the sphere's vz to lift_speed at t=0 in start_gripped mode.
    # Diagnostic for the friction-overshoot hypothesis: if v_rel=0 at t=0,
    # regularized Coulomb can't drive slip, and the overshoot should not
    # appear.  If overshoot still appears with this flag, the hypothesis
    # is wrong and something else is going on.
    warm_start_sphere_vz: bool = False
    # Height at which to spawn the sphere in start_gripped mode.  Chosen so
    # the sphere sits well above where the ground would be even if no_ground
    # is False, so ground contact never becomes relevant during the test.
    gripped_spawn_z: float = 0.20

    @property
    def sphere_start_z(self) -> float:
        """Sphere centre z [m] — rests on the table surface."""
        return self.table_surface_z + self.sphere_radius

    @property
    def robot_base_pos(self) -> tuple[float, float, float]:
        """Robot base world position — directly in front of the table."""
        return (
            self.table_center_x - self.table_hx - self.robot_standoff,
            self.table_center_y,
            self.robot_mount_z,
        )

    @property
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * math.pi * self.sphere_radius ** 3

    @property
    def reach_steps(self):
        return int(self.reach_duration / self.dt)

    @property
    def approach_steps(self):
        return int(self.approach_duration / self.dt)

    @property
    def squeeze_steps(self):
        return int(self.squeeze_duration / self.dt)

    @property
    def lift_steps(self):
        return int(self.lift_duration / self.dt)

    @property
    def hold_steps(self):
        return int(self.hold_duration / self.dt)

    @property
    def total_steps(self):
        if self.start_gripped:
            return self.lift_steps + self.hold_steps
        return (self.reach_steps + self.approach_steps + self.squeeze_steps
                + self.lift_steps + self.hold_steps)


def get_sphere_cfg(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density)
    

def point_pad_cfg(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density)
    return cfg


def cslc_pad_cfg(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        is_cslc=True,
        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    return cfg


def hydro_pad_cfg(p: SceneParams):
    """Build the same articulated-pad scene with hydroelastic contact.

    Both pads AND the sphere need is_hydroelastic=True (PFC requires both
    bodies to carry pressure fields).  kh is the hydroelastic modulus [Pa];
    see SceneParams.kh docstring and section 9 of convo_april_19.md.
    """
    pad_cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        kh=p.kh, is_hydroelastic=True, sdf_max_resolution=p.sdf_resolution)
    return pad_cfg
    # return _build_scene(p, pad_cfg, sphere_cfg=sphere_cfg)