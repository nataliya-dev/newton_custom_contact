from dataclasses import dataclass
import enum
import numpy as np

class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    HOLD = 4


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""
    # SPHERE
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0

    # TABLE
    table_height: float = 0.2
    table_length: float = 0.8
    table_width: float = 0.8
    table_pos: tuple = (0.0, -0.5, 1/2 * table_height)
    table_top_center: tuple = (table_pos[0], table_pos[1], table_pos[2] + 1/2 * table_height)
    
    # ROBOT
    robot_base_pos: tuple = (table_pos[0] - 0.7, table_pos[1], 0.0)
    
    # TASK
    task_offset_approach: tuple = (0.0, 0.0, sphere_radius)
    task_offset_lift: tuple = (0.0, 0.0, 4.0 * sphere_radius)
    task_schedule_time = (
        (TaskType.APPROACH, 2.0),
        (TaskType.REFINE_APPROACH, 2.0),
        (TaskType.GRASP, 2.0),
        (TaskType.LIFT, 2.0),
        (TaskType.HOLD, 2.0),
    )
    penetration_depth = 0.02

    # PADS
    pad_hx: float = 0.01
    pad_hy: float = 0.002
    pad_hz: float = 0.01
    pad_local_z: float = 0.04525   # local z along finger axis — rubber-tip centre [m]
    pad_density: float = 1000.0
    
    # TODO
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

    
    start_gripped: bool = False

    @property
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * np.pi * self.sphere_radius ** 3

    def dump(self):
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        _log(
            f"Sphere: r={self.sphere_radius*1e3:.1f}mm  mass={m*1e3:.1f}g  weight={m*9.81:.3f}N")
        _log(f"Pads:   hx={self.pad_hx*1e3:.0f}mm  hy={self.pad_hy*1e3:.0f}mm  "
             f"hz={self.pad_hz*1e3:.0f}mm  density={self.pad_density:.0f}")
        _log(f"Phases: approach={self.approach_duration}s  squeeze={self.squeeze_duration}s  "
             f"lift={self.lift_duration}s  hold={self.hold_duration}s")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"Drive: ke={self.drive_ke:.0f}  kd={self.drive_kd:.0f}")
        _log(f"Steps: {self.total_steps} total  dt={self.dt*1e3:.2f}ms")