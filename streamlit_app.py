"""Interactive Additive Build Advisor — drop in an STL and step through the thread.

A browser front-end over the *same* Python pipeline the CLI runs
(``abadvisor.pipeline.advise``): upload an STL (or pick a sample part), choose a
process, and walk the part through every stage one at a time — geometry check,
orientation screening, build simulation, the thermal-contraction warpage FEA,
DfAM, the inspection plan, and the release gate — with the real figures and an
interactive 3-D view at each step.

Built as a teaching aid for ES 51 (Computer-Aided Machine Design). Run locally
with ``streamlit run streamlit_app.py``; deployed on Streamlit Community Cloud
straight from this repo (it installs requirements.txt and runs this file).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

# Make the in-repo package importable on Streamlit Cloud (no editable install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st  # noqa: E402

from abadvisor import report, shapes  # noqa: E402
from abadvisor.geometry import Mesh  # noqa: E402
from abadvisor.materials import get_profile, list_profiles  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402

try:  # interactive 3-D is a nice-to-have; degrade gracefully if plotly is absent
    import numpy as np
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:  # pragma: no cover
    _PLOTLY = False

st.set_page_config(page_title="Additive Build Advisor", page_icon="🧩", layout="wide")

_SAMPLES = {
    "gantry_bracket": (lambda: shapes.gantry_bracket(), "examples/tolerances_bracket.json"),
    "calibration_cube": (lambda: shapes.cube(20.0), None),
    "hollow_housing": (lambda: shapes.hollow_housing(), "examples/tolerances_housing.json"),
    "cantilever_benchmark": (lambda: shapes.cantilever_benchmark(), None),
    "tall_standoff": (lambda: shapes.tall_standoff(), None),
}

_STAGES = [
    "1 · Geometry",
    "2 · Orientation",
    "3 · Build simulation",
    "4 · Warpage FEA",
    "5 · Manufacturability (DfAM)",
    "6 · Inspection plan",
    "7 · Release gate",
]

_GATE_UI = {
    "release_to_build": (st.success, "✅ RELEASE TO BUILD"),
    "needs_engineering_review": (st.warning, "🟠 NEEDS ENGINEERING REVIEW"),
    "redesign_required": (st.error, "🔴 REDESIGN REQUIRED"),
}


# --------------------------------------------------------------------------- #
# Pipeline plumbing
# --------------------------------------------------------------------------- #
def _load_tolerances(rel_path):
    import json
    p = os.path.join(os.path.dirname(__file__), rel_path)
    if rel_path and os.path.exists(p):
        return json.loads(open(p).read())
    return None


def _run(mesh=None, stl_path=None, source_label=None, process="fff_pla", grid_n=64, tol=None):
    """Run the advisor and render the per-stage figures into a temp dir."""
    result = advise(mesh=mesh, stl_path=stl_path, source_label=source_label,
                    process=process, tolerance_spec=tol, grid_n=grid_n)
    figdir = tempfile.mkdtemp(prefix="aba_figs_")
    figs = report.render_figures(result, figdir)
    return result, figs


def _mesh3d_input(mesh, name):
    """Interactive 3-D of the input triangle soup."""
    tris = mesh.triangles  # (F, 3, 3)
    verts = tris.reshape(-1, 3)
    n = tris.shape[0]
    i = np.arange(0, 3 * n, 3)
    fig = go.Figure(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i, j=i + 1, k=i + 2,
        color="#9bb7d4", opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2),
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460,
                      scene=dict(aspectmode="data"), title=f"{name} — rotate / zoom")
    return fig


def _mesh3d_warp(fea):
    """Interactive 3-D of the deformed (exaggerated) surface, coloured by warpage."""
    if fea.nodes is None or fea.nodes.shape[1] == 0 or fea.quads.shape[1] == 0:
        return None
    p = fea.nodes                       # (3, N)
    quads = fea.quads                   # (4, M)
    u = fea.u_nodal                     # (3, N)
    mag = fea.mag_nodal                 # (N,)
    maxd = max(float(fea.max_displacement_mm), 1e-9)
    ext = float((p.max(axis=1) - p.min(axis=1)).max())
    scale = 0.10 * ext / maxd           # exaggerate so warpage reads
    dp = p + u * scale
    # split each quad [a,b,c,d] into triangles (a,b,c) and (a,c,d)
    a, b, c, d = quads[0], quads[1], quads[2], quads[3]
    i = np.concatenate([a, a]); j = np.concatenate([b, c]); k = np.concatenate([c, d])
    fig = go.Figure(go.Mesh3d(
        x=dp[0], y=dp[1], z=dp[2], i=i, j=j, k=k,
        intensity=mag, colorscale="Turbo", showscale=True,
        colorbar=dict(title="|u| (mm)"), flatshading=False,
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=460,
                      scene=dict(aspectmode="data"),
                      title=f"Deformed mesh (×{scale:.0f} exaggeration)")
    return fig


# --------------------------------------------------------------------------- #
# Per-stage renderers
# --------------------------------------------------------------------------- #
def _stage_geometry(result, figs):
    rec = result["record"]; part = rec["part"]; wt = part["watertight"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watertight", "yes" if wt.get("is_watertight") else "no")
    c2.metric("Volume", f"{part['volume_mm3']/1000:.2f} cm³")
    c3.metric("Facets", f"{part['n_facets']:,}")
    lo, hi = part["bbox_min_mm"], part["bbox_max_mm"]
    c4.metric("Bounding box", f"{hi[0]-lo[0]:.0f}×{hi[1]-lo[1]:.0f}×{hi[2]-lo[2]:.0f} mm")
    st.caption("The STL is parsed from scratch, normals recomputed from winding, and "
               "watertightness checked before anything downstream trusts the mesh.")
    if _PLOTLY:
        st.plotly_chart(_mesh3d_input(result["mesh"], part["name"]), use_container_width=True)
    elif figs.get("part3d"):
        st.image(figs["part3d"])


def _stage_orientation(result, figs):
    o = result["record"]["design_decision"]["chosen_orientation"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Down-face", str(o.get("label", "")).replace("face ⟂ ", ""))
    c2.metric("Base contact", f"{o.get('base_contact_mm2', 0):.0f} mm²")
    c3.metric("Support volume", f"{o.get('support_volume_mm3', 0):.0f} mm³")
    a, b = st.columns(2)
    if figs.get("orientation"):
        a.image(figs["orientation"], caption="Candidate face-down orientations, scored on support volume")
    if figs.get("part3d"):
        b.image(figs["part3d"], caption="Part in the chosen orientation")
    st.caption("Candidates rest a real flat face on the bed; each is scored on actual support "
               "volume, base-contact area, and build height — no degenerate tilts.")


def _stage_build(result, figs):
    sim = result["record"]["simulation"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Layers", f"{sim['n_layers']:,}")
    c2.metric("Build time", f"{sim['build_time_h']:.2f} h")
    c3.metric("Cost", f"${sim['total_cost_usd']:.2f}")
    a, b = st.columns(2)
    if figs.get("layers"):
        a.image(figs["layers"], caption="Per-layer cross-section vs build height")
    if figs.get("cost_time"):
        b.image(figs["cost_time"], caption="Cost and time breakdown")


def _stage_fea(result, figs):
    fea = result["record"]["distortion_fea"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak warpage", f"{fea['max_distortion_mm']:.3f} mm")
    c2.metric("Elements", f"{fea['elements']:,}")
    vm = fea.get("peak_von_mises_mpa")
    c3.metric("Peak von Mises", f"{vm:.0f} MPa" if vm is not None else "—")
    st.caption("A linear-elastic FEM (scikit-fem hex mesh + SciPy): each element carries a "
               "thermal-contraction eigenstrain (ε* ≈ −α·ΔT, the cooling shrinkage), the first "
               "layer is clamped to the bed, and the solved displacement field is the bed-warpage "
               "— the corner-lift that curls FFF parts off the bed.")
    if _PLOTLY:
        warp = _mesh3d_warp(result["fea"])
        if warp is not None:
            st.plotly_chart(warp, use_container_width=True)
    if figs.get("distortion"):
        st.image(figs["distortion"], caption="Deformed element mesh, coloured by displacement")


def _stage_dfam(result, figs):
    rows = [{"check": f["check"], "severity": f["severity"], "finding": f["message"],
             "recommendation": f["recommendation"]}
            for f in result["record"]["manufacturability"]["findings"]]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Thin walls, support burden, aspect ratio, trapped powder/resin, and the FEA "
               "warpage ratio — severity-ranked, read from the same voxel model.")


def _stage_inspection(result, figs):
    insp = result["record"]["inspection_plan"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tightest tolerance", f"±{insp.get('tightest_tolerance_mm', 0)} mm")
    c2.metric("Requires CMM", str(insp.get("requires_cmm", False)))
    c3.metric("Requires CT", str(insp.get("requires_ct", False)))
    rows = [{"feature": s["feature"], "characteristic": s["characteristic"], "pass if": s["pass_if"],
             "method": s["method"], "equipment": s["equipment"], "capability": s["severity"]}
            for s in insp["steps"]]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No toleranced features supplied — upload a tolerance spec or pick the bracket/housing "
                "sample to see the as-built capability check.")
    st.caption("Each toleranced dimension becomes an inspection step; tolerances below the process's "
               "as-built capability are flagged for finishing (e.g. on the mill).")


def _stage_gate(result, figs):
    gate = result["record"]["gate"]
    fn, label = _GATE_UI.get(gate["decision"], (st.info, gate["decision"].upper()))
    fn(f"**Release gate: {label}**  ·  simulation confidence {gate['confidence']}")
    st.markdown("**Why:**")
    for r in gate["reasons"]:
        st.markdown(f"- {r}")
    h = result["record"]["handoff"]["as_built_context"]
    st.markdown("**Hand-off to the runtime monitoring twin:**")
    st.code(f"machine_id : {h['machine_id']}\n"
            f"part_id    : {h['part_id']}\n"
            f"operation  : {h['operation']}\n"
            f"watch      : {', '.join(h['watch'])}", language="text")
    st.caption("The advisor never silently approves a build. On release, this context is handed to the "
               "companion mini-manufacturing-digital-twin, which watches the part on the machine.")


_RENDER = [_stage_geometry, _stage_orientation, _stage_build, _stage_fea,
           _stage_dfam, _stage_inspection, _stage_gate]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("Additive Build Advisor")
st.markdown(
    "A teaching demo for **ES 51 (Computer-Aided Machine Design)**. Drop in an STL, pick a process, "
    "and step the part through the whole **design → build → inspect** thread — orientation, build "
    "simulation, the thermal-contraction **warpage FEA**, manufacturability, and the **release gate** "
    "— one stage at a time. It runs the same Python pipeline as the command-line tool."
)

with st.sidebar:
    st.header("Part & process")
    src = st.radio("Input", ["Upload an STL", "Use a sample part"], index=1)
    uploaded = st.file_uploader("STL file (binary or ASCII)", type=["stl"]) if src == "Upload an STL" else None
    sample = None
    if src == "Use a sample part":
        sample = st.selectbox("Sample part", list(_SAMPLES), index=0)
    profiles = list_profiles()
    pkey = st.selectbox("Process", [p.key for p in profiles],
                        index=0, format_func=lambda k: get_profile(k).name)
    grid_n = st.select_slider("Voxel resolution", options=[32, 48, 64, 80, 96], value=64)
    go_btn = st.button("Run the advisor ▶", type="primary", use_container_width=True)
    st.caption("FFF is the home process; SLA and SLS are there for comparison. Higher voxel "
               "resolution is more accurate but slower.")

# Run on click; cache the result + figures in session state so stepping is instant.
if go_btn:
    try:
        if src == "Upload an STL":
            if uploaded is None:
                st.sidebar.error("Choose an STL file first.")
                st.stop()
            data = uploaded.getvalue()
            key = hashlib.md5(data + pkey.encode() + str(grid_n).encode()).hexdigest()
            tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
            tmp.write(data); tmp.flush(); tmp.close()
            with st.spinner("Running the pipeline (parse → orient → voxelize → simulate → FEA → DfAM → gate)…"):
                result, figs = _run(stl_path=tmp.name, source_label=uploaded.name, process=pkey, grid_n=grid_n, tol=None)
        else:
            factory, tolpath = _SAMPLES[sample]
            key = f"{sample}-{pkey}-{grid_n}"
            with st.spinner("Running the pipeline…"):
                result, figs = _run(mesh=factory(), source_label=sample, process=pkey,
                                     grid_n=grid_n, tol=_load_tolerances(tolpath))
        st.session_state["run"] = {"result": result, "figs": figs, "key": key}
        st.session_state["step"] = 0
    except Exception as e:  # surface pipeline errors instead of a blank page
        st.error(f"Could not process that part: {e}")

run = st.session_state.get("run")
if not run:
    st.info("⬅️ Pick a sample part (or upload an STL) and press **Run the advisor**.")
    st.stop()

result, figs = run["result"], run["figs"]
step = st.session_state.get("step", 0)

# Stage navigation
st.divider()
nav_l, nav_mid, nav_r = st.columns([1, 3, 1])
with nav_l:
    if st.button("◀ Back", disabled=step == 0, use_container_width=True):
        st.session_state["step"] = max(0, step - 1); st.rerun()
with nav_r:
    if st.button("Next ▶", disabled=step == len(_STAGES) - 1, use_container_width=True):
        st.session_state["step"] = min(len(_STAGES) - 1, step + 1); st.rerun()
with nav_mid:
    st.progress((step + 1) / len(_STAGES), text=f"Stage {step + 1} of {len(_STAGES)}")

st.subheader(_STAGES[step])
_RENDER[step](result, figs)

# quick stage jump
st.divider()
cols = st.columns(len(_STAGES))
for idx, col in enumerate(cols):
    if col.button(_STAGES[idx].split(" · ")[0], use_container_width=True, key=f"jump{idx}"):
        st.session_state["step"] = idx; st.rerun()
