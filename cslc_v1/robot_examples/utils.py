import numpy as np

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