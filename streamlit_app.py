"""Additive Build Advisor — a guided, interactive front-end over the pipeline.

Drop in an STL (or pick a sample part), choose a process, and walk the part
through the whole *design -> build -> inspect -> decide* thread one stage at a
time: geometry + watertight check, build-orientation screening, the layer-by-layer
build simulation, the thermal-contraction warpage FEA, DfAM manufacturability
checks, the first-article inspection plan, and the release gate.

This is only a front-end. Every number, figure, and verdict shown here comes from
the *same* Python pipeline the command-line tool runs (``abadvisor.pipeline.advise``);
nothing on this page is faked or hard-coded. It is a compact teaching prototype
that makes the engineering decisions legible, not a production build processor.

Run locally with ``streamlit run streamlit_app.py``; it also deploys on Streamlit
Community Cloud straight from this repo (it installs requirements.txt and runs
this file).
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import sys
import tempfile

# Make the in-repo package importable on Streamlit Cloud (no editable install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np  # noqa: E402  (hard dependency of the pipeline)
import streamlit as st  # noqa: E402

from abadvisor import report, shapes  # noqa: E402
from abadvisor.materials import get_profile, list_profiles  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402

try:  # interactive 3-D is a nice-to-have; degrade to matplotlib if plotly is absent
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:  # pragma: no cover
    _PLOTLY = False

# Escape hatch for verifying the matplotlib fallback without uninstalling plotly.
if os.environ.get("ABA_FORCE_NO_PLOTLY") == "1":  # pragma: no cover
    _PLOTLY = False

st.set_page_config(
    page_title="Additive Build Advisor",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Configuration / copy
# --------------------------------------------------------------------------- #
_SAMPLES = {
    "gantry_bracket": (lambda: shapes.gantry_bracket(), "examples/tolerances_bracket.json",
                       "L-bracket with a top-flange overhang — a clean release with a couple of tight features."),
    "calibration_cube": (lambda: shapes.cube(20.0), None,
                         "The canonical 20 mm cube — the simplest possible clean build."),
    "hollow_housing": (lambda: shapes.hollow_housing(), "examples/tolerances_housing.json",
                       "A sealed box with a fully enclosed cavity — trapped material, a redesign case."),
    "cantilever_benchmark": (lambda: shapes.cantilever_benchmark(), None,
                             "A long, thin, flat bar — the classic warp-prone geometry for the FEA."),
    "tall_standoff": (lambda: shapes.tall_standoff(), None,
                      "A tall slender cylinder — exercises the aspect-ratio / stability check."),
}

# Phases group the seven stages into the design -> build -> inspect -> decide arc.
_PHASES = [
    ("Design", "#2b6cb0"),
    ("Build", "#0f766e"),
    ("Inspect", "#6d28d9"),
    ("Decide", "#b45309"),
]

# One entry per stage: short ribbon label, full title, phase index, and the
# plain-language "what just happened / why it matters" copy.
_STAGES = [
    dict(short="Geometry", title="Geometry & watertight check", phase=0,
         what="The STL is parsed from scratch, its facet normals are recomputed from the "
              "triangle winding, and the mesh is tested for watertightness.",
         why="Everything downstream trusts this mesh. A leaky or non-manifold model makes the "
             "inside/outside test unreliable, so it is caught here before it can corrupt the "
             "volume, support, and warpage estimates."),
    dict(short="Orientation", title="Build-orientation screening", phase=0,
         what="Candidate \"rest a flat face on the bed\" orientations are generated from the part's "
              "own flat faces and scored on support volume, base-contact area, and build height.",
         why="Orientation is the highest-leverage decision in additive manufacturing: it sets how much "
             "support you print, how well the part sticks to the bed, and how tall (and slow) the build is."),
    dict(short="Build sim", title="Layer-by-layer build simulation", phase=1,
         what="The oriented part is voxelized and turned into a build: layer count, per-layer "
              "cross-section, support material, time, and cost.",
         why="This is the estimate you check before committing machine time — and the voxel volume is "
             "cross-checked against the analytic mesh volume so you know the discretization can be trusted."),
    dict(short="Warpage FEA", title="Thermal-contraction warpage FEA", phase=1,
         what="A linear-elastic finite-element solve (scikit-fem) applies the material's cooling "
              "shrinkage as a uniform eigenstrain and clamps the first layer to the bed.",
         why="The solved displacement field is the corner-lift that curls parts off the bed as they "
             "cool — the dominant geometric defect in FFF, worst on large flat footprints."),
    dict(short="DfAM", title="Manufacturability (DfAM) checks", phase=2,
         what="Design-for-additive checks read the same voxel model: thin walls, support burden, "
              "aspect ratio, trapped material, and the FEA warpage ratio.",
         why="Each finding is ranked by severity; the worst one drives the release gate. This is the "
             "manufacturability review a build-prep engineer runs before a part is cleared."),
    dict(short="Inspection", title="First-article inspection plan", phase=2,
         what="Each toleranced dimension becomes a measurement step with a method and equipment, and "
              "the tolerance is checked against the process's as-built capability.",
         why="Tolerances tighter than the process can hold as-built are flagged for a finishing step, "
             "so you plan the post-machining instead of inspecting to a guaranteed failure."),
    dict(short="Release gate", title="Release gate & hand-off", phase=3,
         what="The record is assembled and a gate is applied: release to build, needs engineering "
              "review, or redesign required — with the reasons attached.",
         why="The advisor never silently approves a build. On release, a machine-readable context is "
             "handed to a runtime monitoring twin; anything uncertain is routed to a human."),
]

_GATE_META = {
    "release_to_build": ("rel", "Release to build", "✅"),
    "needs_engineering_review": ("rev", "Needs engineering review", "🟠"),
    "redesign_required": ("red", "Redesign required", "🔴"),
}

# Reliability caps for uploads (the voxelizer is pure-Python, so very large or
# empty meshes are refused / warned about rather than allowed to hang the app).
_MAX_UPLOAD_MB = 25
_MAX_FACETS = 60000
_WARN_FACETS = 8000
_GRID_OPTIONS = [32, 48, 64, 80, 96]


# --------------------------------------------------------------------------- #
# Design system (custom CSS)
# --------------------------------------------------------------------------- #
_CSS = """
<style>
:root{
  --bg:#eef1f6; --panel:#ffffff; --ink:#1a2230; --ink2:#3a4658; --muted:#67728a;
  --line:#e3e8f0; --accent:#2563eb; --accent-ink:#1e40af;
  --ok:#1b7f3b; --info:#2c6fbb; --warn:#b8860b; --crit:#b22222;
  --rel:#1b7f3b; --rev:#b8860b; --red:#b22222;
}
html, body, [class*="css"]{
  font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{ background:
  radial-gradient(1200px 500px at 15% -10%, #f7fafe 0%, rgba(247,250,254,0) 60%),
  linear-gradient(180deg,#eef2f7 0%, #e9edf4 100%); }
/* tighten the default top padding so the hero sits high */
.block-container{ padding-top:1.2rem; padding-bottom:3rem; max-width:1200px; }
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }

/* ---- entrance animation applied to generated content + heavy widgets ---- */
@keyframes abaFade{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;} }
.aba-anim,
[data-testid="stPlotlyChart"],
[data-testid="stImage"]{ animation:abaFade .38s cubic-bezier(.22,.61,.36,1); }

/* ---- hero ---- */
.aba-hero{
  background:linear-gradient(120deg,#0f2540 0%, #1e4e8c 55%, #2b6cb0 100%);
  color:#eaf1fb; border-radius:16px; padding:22px 26px; margin:2px 0 14px;
  box-shadow:0 10px 30px rgba(15,37,64,.18); position:relative; overflow:hidden;
}
.aba-hero::after{ content:""; position:absolute; inset:0;
  background-image:linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size:26px 26px; opacity:.5; pointer-events:none; }
.aba-hero h1{ margin:0; font-size:27px; letter-spacing:-.4px; font-weight:800; position:relative; }
.aba-hero p{ margin:7px 0 0; color:#cfe0f4; font-size:15px; max-width:900px; position:relative; }
.aba-badges{ margin-top:12px; position:relative; }
.aba-badge{ display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  color:#eaf1fb; font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px; margin:0 6px 6px 0; }

/* ---- cards / panels ---- */
.aba-card{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.aba-eyebrow{ font-size:12px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
.aba-h{ font-size:20px; font-weight:800; color:var(--ink); margin:2px 0 2px; letter-spacing:-.2px; }

/* ---- stat cards ---- */
.stat-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:4px 0 2px; }
.stat{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px;
  box-shadow:0 1px 2px rgba(16,24,40,.04); }
.stat-k{ color:var(--muted); font-size:12px; font-weight:600; letter-spacing:.02em; }
.stat-v{ color:var(--ink); font-size:23px; font-weight:800; margin-top:3px; line-height:1.1;
  font-variant-numeric:tabular-nums; }
.stat-s{ color:var(--muted); font-size:12px; margin-top:2px; }
.stat.accent{ border-color:#c9dcf7; background:linear-gradient(180deg,#f5f9ff,#ffffff); }

/* ---- callout (stage explanation) ---- */
.callout{ display:flex; gap:12px; align-items:flex-start; background:var(--panel); border:1px solid var(--line);
  border-left:4px solid var(--accent); border-radius:12px; padding:13px 15px; margin:2px 0 8px; }
.ic-badge{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px;
  border-radius:50%; background:var(--accent); color:#fff; font-weight:800; font-style:italic;
  font-size:13px; flex:0 0 auto; margin-top:2px; }
.ic-badge.warn{ background:var(--warn); font-style:normal; }
.callout .tx{ font-size:14.5px; color:var(--ink2); }
.callout .tx b{ color:var(--ink); }

/* ---- severity chips + tables ---- */
.chip{ display:inline-block; color:#fff; font-size:11px; font-weight:700; letter-spacing:.03em;
  text-transform:uppercase; padding:2px 9px; border-radius:999px; }
.sev-ok{ background:var(--ok);} .sev-info{ background:var(--info);}
.sev-warning{ background:var(--warn);} .sev-critical{ background:var(--crit);}
.aba-table{ width:100%; border-collapse:separate; border-spacing:0; background:var(--panel);
  border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:13.5px; }
.aba-table th{ text-align:left; background:#f4f7fb; color:#556; font-size:11px; font-weight:700;
  text-transform:uppercase; letter-spacing:.05em; padding:9px 12px; border-bottom:1px solid var(--line); }
.aba-table td{ padding:10px 12px; border-bottom:1px solid #eef2f7; color:var(--ink2); vertical-align:top; }
.aba-table tr:last-child td{ border-bottom:none; }
.aba-table tr:hover td{ background:#f9fbfe; }
.aba-table td b{ color:var(--ink); }

/* ---- verdict banner ---- */
.verdict{ border-radius:16px; padding:18px 20px; color:#fff; box-shadow:0 10px 24px rgba(16,24,40,.14);
  display:flex; align-items:center; gap:16px; }
.verdict .vic{ font-size:38px; }
.verdict .vt{ font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; opacity:.9; }
.verdict .vh{ font-size:26px; font-weight:800; letter-spacing:-.3px; margin-top:1px; }
.v-rel{ background:linear-gradient(120deg,#12703a,#1b7f3b);}
.v-rev{ background:linear-gradient(120deg,#9a6b06,#b8860b);}
.v-red{ background:linear-gradient(120deg,#8f1c1c,#b22222);}
.pulse{ animation:abaPulse 2.4s ease-in-out infinite; }
@keyframes abaPulse{ 0%,100%{ transform:scale(1); } 50%{ transform:scale(1.09); } }

/* ---- confidence meter ---- */
.meter{ height:9px; background:#e7ecf3; border-radius:999px; overflow:hidden; margin-top:6px; }
.meter > span{ display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg,#2b6cb0,#2563eb); }

/* ---- reasons list ---- */
.reasons{ margin:0; padding-left:20px; } .reasons li{ margin:5px 0; color:var(--ink2); font-size:14px; }

/* ---- phase ribbon labels ---- */
.phase-row{ display:flex; gap:8px; margin:4px 0 2px; flex-wrap:wrap; }
.phase-tag{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  padding:3px 10px; border-radius:999px; border:1px solid var(--line); background:#fff; color:var(--muted); }

/* ---- native buttons tuned to read as ribbon chips ---- */
.stButton>button{ border-radius:10px; font-weight:600; border:1px solid var(--line);
  transition:transform .08s ease, box-shadow .12s ease; }
.stButton>button:hover{ transform:translateY(-1px); box-shadow:0 3px 10px rgba(16,24,40,.10); }
.stDownloadButton>button{ border-radius:10px; font-weight:600; }

/* ---- misc ---- */
.hand{ background:#0f1b2d; color:#cfe0f4; border-radius:12px; padding:14px 16px; font-size:13px;
  font-family:"SFMono-Regular",ui-monospace,Menlo,Consolas,monospace; line-height:1.7; overflow-x:auto; }
.hand .k{ color:#7fb0ec; } .small-note{ color:var(--muted); font-size:12.5px; margin-top:8px; }
.filmstrip{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
.film{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px;
  border-top:4px solid var(--accent); }
.film .fp{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
.film .fh{ font-size:15px; font-weight:800; color:var(--ink); margin:3px 0 6px; }
.film .fl{ font-size:13px; color:var(--ink2); }
</style>
"""


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #
def _esc(x) -> str:
    return _html.escape(str(x))


def _md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def _stat_grid(cards) -> str:
    """cards: list of (label, value, sub_or_None, accent_bool)."""
    cells = []
    for label, value, sub, accent in cards:
        sub_html = f'<div class="stat-s">{_esc(sub)}</div>' if sub else ""
        cls = "stat accent" if accent else "stat"
        cells.append(f'<div class="{cls}"><div class="stat-k">{_esc(label)}</div>'
                     f'<div class="stat-v">{_esc(value)}</div>{sub_html}</div>')
    return f'<div class="stat-grid aba-anim">{"".join(cells)}</div>'


def _callout(what: str, why: str) -> str:
    return (f'<div class="callout aba-anim"><span class="ic-badge">i</span><div class="tx">'
            f'<b>What just happened.</b> {_esc(what)}<br><b>Why it matters.</b> {_esc(why)}'
            f'</div></div>')


def _findings_table(headers, rows, sev_index) -> str:
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            if i == sev_index:
                tds.append(f'<td><span class="chip sev-{_esc(c)}">{_esc(c)}</span></td>')
            else:
                tds.append(f"<td>{_esc(c)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="aba-anim"><table class="aba-table"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# --------------------------------------------------------------------------- #
# Pipeline plumbing (unchanged science; just orchestration + caching)
# --------------------------------------------------------------------------- #
def _load_tolerances(rel_path):
    if not rel_path:
        return None
    p = os.path.join(os.path.dirname(__file__), rel_path)
    if os.path.exists(p):
        with open(p) as fh:
            return json.loads(fh.read())
    return None


def _decimate_build(grid, target=4500):
    """Precompute a light voxel point cloud of the build for the layer explorer.

    The full grid can be hundreds of thousands of cells; the 3-D layer view only
    needs an illustrative decimation, so we stride it down to ~``target`` points.
    """
    occ = grid.occ
    nx, ny, nz = occ.shape
    pitch = grid.pitch
    s = max(1, int(np.ceil(max(nx, ny, nz) / 40.0)))
    while True:
        docc = occ[::s, ::s, ::s]
        idx = np.argwhere(docc)
        if idx.shape[0] <= target or s >= max(nx, ny, nz):
            break
        s += 1
    if idx.shape[0] == 0:
        return {"x": np.zeros(0), "y": np.zeros(0), "z": np.zeros(0), "zk": np.zeros(0, int),
                "s": s, "nz": nz}
    zk = idx[:, 2] * s  # approximate original z-index of each decimated point
    return {
        "x": idx[:, 0] * s * pitch, "y": idx[:, 1] * s * pitch, "z": idx[:, 2] * s * pitch,
        "zk": zk, "s": s, "nz": nz,
    }


def _run(mesh=None, stl_path=None, source_label=None, process="fff_pla", grid_n=64, tol=None):
    """Run the real advisor, render the report figures, and self-contained HTML."""
    result = advise(mesh=mesh, stl_path=stl_path, source_label=source_label,
                    process=process, tolerance_spec=tol, grid_n=grid_n)
    figdir = tempfile.mkdtemp(prefix="aba_figs_")
    figs = report.render_figures(result, figdir)
    try:
        html_path = report.render_html(result, figdir, embed=True)
        report_html = open(html_path, "rb").read()
    except Exception:
        report_html = None
    decim = _decimate_build(result["grid"])
    return result, figs, report_html, decim


# --------------------------------------------------------------------------- #
# Plotly builders (graceful no-op when plotly is unavailable)
# --------------------------------------------------------------------------- #
_PART_BLUE = "#9bb7d4"
_OVERHANG = "#e8833a"


def _bed_trace(extent_xy, z=0.0):
    r = 0.7 * extent_xy + 4.0
    return go.Mesh3d(
        x=[-r, r, r, -r], y=[-r, -r, r, r], z=[z, z, z, z],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#b9c4d4", opacity=0.28, hoverinfo="skip", showscale=False, name="bed",
    )


def _mesh3d_tris(tris, color, opacity=1.0, name=None):
    verts = tris.reshape(-1, 3)
    n = tris.shape[0]
    i = np.arange(0, 3 * n, 3)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=i, j=i + 1, k=i + 2,
        color=color, opacity=opacity, flatshading=True, name=name,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.22, roughness=0.55),
        lightposition=dict(x=120, y=200, z=300),
    )


def _scene(height=470, title=None):
    return dict(
        margin=dict(l=0, r=0, t=30 if title else 0, b=0), height=height,
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)",
                   zaxis_title="z (mm)",
                   xaxis=dict(backgroundcolor="#f4f7fb", gridcolor="#e3e8f0"),
                   yaxis=dict(backgroundcolor="#f4f7fb", gridcolor="#e3e8f0"),
                   zaxis=dict(backgroundcolor="#eef2f7", gridcolor="#e3e8f0")),
        paper_bgcolor="rgba(0,0,0,0)", title=title,
        title_font=dict(size=13, color="#3a4658"),
    )


def _fig_input(mesh, name):
    fig = go.Figure(_mesh3d_tris(mesh.triangles, _PART_BLUE, name=name))
    fig.update_layout(**_scene(470))
    return fig


def _fig_orientation(base_mesh, rotation, self_support_angle, show_overhangs=True):
    oriented = base_mesh.transformed(rotation).dropped_to_plate()
    tris = oriented.triangles
    ext = oriented.extents
    traces = [_bed_trace(float(max(ext[0], ext[1])))]
    if show_overhangs:
        mask = oriented.overhang_mask(self_support_angle)
    else:
        mask = np.zeros(tris.shape[0], dtype=bool)
    if (~mask).any():
        traces.append(_mesh3d_tris(tris[~mask], _PART_BLUE, name="part"))
    if mask.any():
        traces.append(_mesh3d_tris(tris[mask], _OVERHANG, name="needs support"))
    fig = go.Figure(traces)
    fig.update_layout(**_scene(470))
    fig.update_layout(showlegend=False)
    return fig, int(mask.sum())


def _fig_warp(fea, exagg, show_ghost=True):
    if fea.nodes is None or fea.nodes.shape[1] == 0 or fea.quads.shape[1] == 0:
        return None
    p = fea.nodes
    quads = fea.quads
    u = fea.u_nodal
    mag = fea.mag_nodal
    a, b, c, d = quads[0], quads[1], quads[2], quads[3]
    i = np.concatenate([a, a]); j = np.concatenate([b, c]); k = np.concatenate([c, d])
    dp = p + u * float(exagg)
    traces = []
    if show_ghost and exagg > 0:
        traces.append(go.Mesh3d(x=p[0], y=p[1], z=p[2], i=i, j=j, k=k,
                                color="#c3ccd8", opacity=0.14, hoverinfo="skip",
                                showscale=False, name="undeformed"))
    traces.append(go.Mesh3d(
        x=dp[0], y=dp[1], z=dp[2], i=i, j=j, k=k,
        intensity=mag, colorscale="Turbo", showscale=True,
        colorbar=dict(title="|u| (mm)", thickness=14, len=0.72),
        flatshading=False, name="warped",
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.15),
    ))
    fig = go.Figure(traces)
    fig.update_layout(**_scene(500))
    fig.update_layout(showlegend=False)
    return fig


def _fig_build3d(decim, k_full):
    x, y, z, zk = decim["x"], decim["y"], decim["z"], decim["zk"]
    if x.shape[0] == 0:
        return None
    built = zk <= k_full
    fig = go.Figure()
    if (~built).any():
        fig.add_trace(go.Scatter3d(
            x=x[~built], y=y[~built], z=z[~built], mode="markers",
            marker=dict(size=2.4, color="#d5dbe4", opacity=0.35), hoverinfo="skip", name="remaining"))
    if built.any():
        fig.add_trace(go.Scatter3d(
            x=x[built], y=y[built], z=z[built], mode="markers",
            marker=dict(size=3.4, color=z[built], colorscale="Viridis", opacity=0.95),
            hoverinfo="skip", name="printed"))
    fig.update_layout(**_scene(430))
    fig.update_layout(showlegend=False)
    return fig


def _fig_cross_section(occ, k, pitch):
    layer = occ[:, :, k].astype(int)
    fig = go.Figure(go.Heatmap(
        z=layer.T, colorscale=[[0, "rgba(0,0,0,0)"], [1, "#2563eb"]], showscale=False,
        x=np.arange(layer.shape[0]) * pitch, y=np.arange(layer.shape[1]) * pitch, hoverinfo="skip"))
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), height=430,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f4f7fb",
                      xaxis=dict(title="x (mm)", showgrid=False),
                      yaxis=dict(title="y (mm)", scaleanchor="x", scaleratio=1, showgrid=False))
    return fig


def _fig_layer_profile(z_centers, area_mm2, k):
    z = np.asarray(z_centers); area = np.asarray(area_mm2)
    built = z <= z[k]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=area, y=z, mode="lines", line=dict(color="#94a3b8", width=1.5),
                             hoverinfo="skip", name="profile"))
    fig.add_trace(go.Scatter(x=np.where(built, area, 0), y=z, mode="lines", fill="tozerox",
                             line=dict(color="#2563eb", width=1.5), fillcolor="rgba(37,99,235,.20)",
                             hoverinfo="skip", name="printed"))
    fig.add_trace(go.Scatter(x=[area[k]], y=[z[k]], mode="markers",
                             marker=dict(size=10, color="#1e40af", line=dict(color="#fff", width=1.5)),
                             hoverinfo="skip", name="current"))
    fig.update_layout(margin=dict(l=0, r=6, t=6, b=0), height=430, showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f4f7fb",
                      xaxis=dict(title="cross-section area (mm²)", gridcolor="#e3e8f0"),
                      yaxis=dict(title="build height z (mm)", gridcolor="#e3e8f0"))
    return fig


# --------------------------------------------------------------------------- #
# Per-stage renderers
# --------------------------------------------------------------------------- #
def _report_fig_expander(figs, key, label):
    if figs.get(key):
        with st.expander(f"Report figure — {label}"):
            st.image(figs[key], use_container_width=True)


def _stage_geometry(result, figs, decim):
    rec = result["record"]; part = rec["part"]; wt = part["watertight"]
    lo, hi = part["bbox_min_mm"], part["bbox_max_mm"]
    dims = f"{hi[0]-lo[0]:.0f} × {hi[1]-lo[1]:.0f} × {hi[2]-lo[2]:.0f} mm"
    watertight = wt.get("is_watertight")
    _md(_stat_grid([
        ("Watertight", "Yes" if watertight else "No",
         "manifold mesh" if watertight else "open/non-manifold", bool(watertight)),
        ("Volume", f"{part['volume_mm3']/1000:.2f} cm³", "from the divergence theorem", False),
        ("Facets", f"{part['n_facets']:,}", "triangles parsed", False),
        ("Bounding box", dims, "x × y × z", False),
    ]))
    if not watertight:
        _md('<div class="callout aba-anim" style="border-left-color:var(--warn)">'
            '<span class="ic-badge warn">!</span><div class="tx"><b>Heads up.</b> This mesh is not watertight, so '
            'the inside/outside test is less reliable and simulation confidence is reduced downstream. '
            'The pipeline still runs; the release gate accounts for it.</div></div>')
    if _PLOTLY:
        st.plotly_chart(_fig_input(result["mesh"], part["name"]), use_container_width=True,
                        config={"displaylogo": False})
        st.caption("Rotate, zoom, and pan — this is the raw triangle soup exactly as parsed from the STL.")
    else:
        _report_fig_expander(figs, "part3d", "part")


def _stage_orientation(result, figs, decim):
    orientation = result["orientation"]
    cands = orientation["candidates"]
    profile = result["profile"]
    rec = result["record"]
    chosen = rec["design_decision"]["chosen_orientation"]

    _md(_stat_grid([
        ("Chosen down-face", str(chosen.get("label", "")).replace("face ⟂ ", ""), "rests on the bed", True),
        ("Base contact", f"{chosen.get('base_contact_mm2', 0):.0f} mm²",
         f"{chosen.get('contact_fraction', 0)*100:.0f}% of footprint", False),
        ("Support volume", f"{chosen.get('support_volume_mm3', 0):.0f} mm³", "material to remove", False),
        ("Build height", f"{chosen.get('height_mm', 0):.1f} mm", "drives build time", False),
    ]))

    if _PLOTLY and len(cands) > 0:
        if len(cands) > 1:
            labels = [f"#{idx+1}  {c.label.replace('face ⟂ ', '')}"
                      + ("   ★ chosen" if idx == 0 else "") for idx, c in enumerate(cands)]
            sel = st.select_slider("Compare candidate orientations (ranked best → worst)",
                                   options=list(range(len(cands))),
                                   value=0, format_func=lambda i: labels[i], key="orient_sel")
        else:
            sel = 0
            st.caption("This part has a single distinct rest-on-face orientation.")
        show_ov = st.checkbox("Highlight facets that would need support", value=True, key="orient_ov")
        cand = cands[sel]
        fig, n_over = _fig_orientation(result["mesh"], cand.rotation,
                                       profile.self_support_angle_deg, show_overhangs=show_ov)
        cc = st.columns([3, 2])
        with cc[0]:
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            legend = "blue = part"
            if show_ov:
                legend += ",  orange = down-facing overhang needing support" if n_over else ",  no support-needing overhangs"
            st.caption(f"Candidate #{sel+1} on the bed — {legend}.")
        with cc[1]:
            fits = "fits" if cand.fits_build_volume else "does NOT fit"
            _md(_stat_grid([
                ("Rank", f"#{sel+1} of {len(cands)}", "★ chosen" if sel == 0 else "alternative", sel == 0),
                ("Score", f"{cand.score:.3f}", "lower is better", False),
                ("Support", f"{cand.support_volume_mm3:.0f} mm³", None, False),
                ("Contact", f"{cand.base_contact_mm2:.0f} mm²", f"{cand.contact_fraction*100:.0f}% footprint", False),
                ("Height", f"{cand.height_mm:.1f} mm", None, False),
                ("Build volume", fits, None, not cand.fits_build_volume),
            ]))
        # ranking table across candidates
        rows = [[f"#{i+1}", c.label.replace("face ⟂ ", ""), f"{c.support_volume_mm3:.0f}",
                 f"{c.base_contact_mm2:.0f}", f"{c.height_mm:.1f}", f"{c.score:.3f}"]
                for i, c in enumerate(cands)]
        _md(_findings_table(["rank", "down-face", "support mm³", "contact mm²", "height mm", "score"],
                            rows, sev_index=-1))
    else:
        _report_fig_expander(figs, "orientation", "orientation screening")
        _report_fig_expander(figs, "part3d", "chosen orientation")


def _stage_build(result, figs, decim):
    sim = result["record"]["simulation"]
    grid = result["grid"]
    _md(_stat_grid([
        ("Layers", f"{sim['n_layers']:,}", None, True),
        ("Build time", f"{sim['build_time_h']:.2f} h", "deposition + overhead", False),
        ("Cost", f"${sim['total_cost_usd']:.2f}", "material + machine", False),
        ("Support", f"{sim['support_material_cm3']:.2f} cm³", "to remove", False),
        ("Volume check", f"{sim['volume_error_pct']:+.2f}%", "voxel vs analytic", False),
    ]))
    if _PLOTLY:
        occ = grid.occ
        nz = occ.shape[2]
        zc = grid.z_centers()
        area = grid.layer_area_mm2()
        # default to the topmost *occupied* slice (the last grid layer is padding)
        default_k = int(np.max(np.nonzero(area > 0))) if np.any(area > 0) else nz - 1
        if nz > 1:
            k = st.slider("Scrub the build layer by layer", 0, nz - 1, default_k, key="build_layer")
        else:
            k = 0  # single-slice part: nothing to scrub
        pct = 100.0 * (k + 1) / nz
        _md(_stat_grid([
            ("Layer", f"{k+1} / {nz}", "voxel slices", True),
            ("Height built", f"{zc[k]:.1f} mm", None, False),
            ("Progress", f"{pct:.0f}%", None, False),
            ("This layer", f"{area[k]:.0f} mm²", "cross-section", False),
        ]))
        cc = st.columns(2)
        with cc[0]:
            f3 = _fig_build3d(decim, k)
            if f3 is not None:
                st.plotly_chart(f3, use_container_width=True, config={"displaylogo": False})
                st.caption("Printed so far (colored by height) vs. remaining (grey). Illustrative voxel resolution.")
        with cc[1]:
            st.plotly_chart(_fig_cross_section(occ, k, grid.pitch), use_container_width=True,
                            config={"displaylogo": False})
            st.caption(f"Cross-section being deposited at z = {zc[k]:.1f} mm.")
        st.plotly_chart(_fig_layer_profile(zc, area, k), use_container_width=True,
                        config={"displaylogo": False})
    else:
        _report_fig_expander(figs, "layers", "layer profile")
    _report_fig_expander(figs, "cost_time", "cost & time breakdown")


def _stage_fea(result, figs, decim):
    fea_rec = result["record"]["distortion_fea"]
    fea = result["fea"]
    vm = fea_rec.get("peak_von_mises_mpa")
    _md(_stat_grid([
        ("Peak warpage", f"{fea_rec['max_distortion_mm']:.3f} mm", "corner-lift off the bed", True),
        ("Mean warpage", f"{fea_rec['mean_distortion_mm']:.3f} mm", None, False),
        ("Elements", f"{fea_rec['elements']:,}", "hex mesh", False),
        ("Peak von Mises", f"{vm:.0f} MPa" if vm is not None else "—", "linear-elastic", False),
    ]))
    if _PLOTLY and fea.nodes is not None and fea.nodes.shape[1] > 0:
        maxd = max(float(fea.max_displacement_mm), 1e-9)
        ext = float((fea.nodes.max(axis=1) - fea.nodes.min(axis=1)).max())
        auto = max(1, min(300, int(round(0.10 * ext / maxd))))  # cap so the range stays sane
        cc = st.columns([3, 1])
        with cc[1]:
            exagg = st.slider("Exaggeration ×", 0, int(auto * 3), int(auto), key="warp_exagg",
                              help="True scale is ×1 (warpage is sub-millimetre). Scale it up so the "
                                   "corner-lift is visible.")
            ghost = st.checkbox("Show undeformed", value=True, key="warp_ghost")
        with cc[0]:
            fig = _fig_warp(fea, exagg, show_ghost=ghost)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        note = "true scale (warpage is tiny)" if exagg <= 1 else f"deformation exaggerated ×{exagg} for visibility"
        st.caption(f"Deformed mesh colored by displacement magnitude — {note}. Near zero at the bed-clamped "
                   "base, rising toward the free corners.")
    else:
        _report_fig_expander(figs, "distortion", "warpage FEA")


def _stage_dfam(result, figs, decim):
    dfam = result["record"]["manufacturability"]
    findings = dfam["findings"]
    n_crit = dfam.get("n_critical", 0); n_warn = dfam.get("n_warning", 0)
    _md(_stat_grid([
        ("Worst severity", dfam.get("worst_severity", "ok").upper(), None,
         dfam.get("worst_severity") in ("warning", "critical")),
        ("Critical", str(n_crit), None, n_crit > 0),
        ("Warnings", str(n_warn), None, n_warn > 0),
        ("Checks run", str(len(findings)), None, False),
    ]))
    rows = [[f["check"].replace("_", " "), f["severity"], f["message"], f["recommendation"]]
            for f in findings]
    _md(_findings_table(["check", "severity", "finding", "recommendation"], rows, sev_index=1))


def _stage_inspection(result, figs, decim):
    insp = result["record"]["inspection_plan"]
    steps = insp["steps"]
    tightest = insp.get("tightest_tolerance_mm")
    _md(_stat_grid([
        ("Inspection steps", str(insp.get("n_steps", len(steps))), None, True),
        ("Tightest tolerance", f"±{tightest} mm" if tightest is not None else "—", None, False),
        ("Requires CMM", "Yes" if insp.get("requires_cmm") else "No", None, bool(insp.get("requires_cmm"))),
        ("Capability flags", str(insp.get("n_capability_flags", 0)), "need finishing",
         insp.get("n_capability_flags", 0) > 0),
    ]))
    if steps:
        rows = [[s["feature"], s["characteristic"], s["pass_if"], s["severity"], s["equipment"]]
                for s in steps]
        _md(_findings_table(["feature", "characteristic", "pass if", "capability", "equipment"],
                            rows, sev_index=3))
    else:
        st.info("No toleranced features were supplied. Pick the bracket or housing sample (they carry "
                "tolerance specs), or upload a part, to see the as-built capability check.")


def _stage_gate(result, figs, decim):
    rec = result["record"]
    _verdict(rec)
    h = rec["handoff"]["as_built_context"]
    st.markdown("**Hand-off to the runtime monitoring twin**")
    _md('<div class="hand aba-anim">'
        f'<span class="k">machine_id</span> : {_esc(h["machine_id"])}<br>'
        f'<span class="k">part_id</span>    : {_esc(h["part_id"])}<br>'
        f'<span class="k">operation</span>  : {_esc(h["operation"])}<br>'
        f'<span class="k">expected</span>   : {_esc(h["expected_layers"])} layers / {_esc(h["expected_build_time_h"])} h<br>'
        f'<span class="k">watch</span>      : {_esc(", ".join(h["watch"]))}</div>')
    _md('<div class="small-note">On release, this context is handed to the companion runtime monitoring '
        'twin, which watches the part on the machine. The advisor never silently approves a build.</div>')
    _downloads(rec, result)


def _verdict(rec):
    gate = rec["gate"]
    klass, label, icon = _GATE_META.get(gate["decision"], ("rev", gate["decision"], "•"))
    conf = float(gate.get("confidence", 0))
    _md(f'<div class="verdict v-{klass} aba-anim"><div class="vic pulse">{icon}</div>'
        f'<div><div class="vt">Release gate · confidence {conf:.2f}</div>'
        f'<div class="vh">{_esc(label)}</div>'
        f'<div class="meter" style="max-width:320px"><span style="width:{conf*100:.0f}%"></span></div>'
        f'</div></div>')
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in gate.get("reasons", []))
    _md(f'<div class="aba-card aba-anim" style="margin-top:12px"><div class="aba-eyebrow" '
        f'style="color:var(--muted)">Reasons</div><ul class="reasons">{reasons}</ul></div>')


def _downloads(rec, result):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ Build record (JSON)", data=json.dumps(rec, indent=2),
                           file_name=f"{rec['part']['name']}_build_record.json",
                           mime="application/json", use_container_width=True)
    with c2:
        rh = st.session_state.get("run", {}).get("report_html")
        if rh:
            st.download_button("⬇ Full report (HTML)", data=rh,
                               file_name=f"{rec['part']['name']}_report.html",
                               mime="text/html", use_container_width=True)


_RENDER = [_stage_geometry, _stage_orientation, _stage_build, _stage_fea,
           _stage_dfam, _stage_inspection, _stage_gate]


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _render_overview(result, figs, decim):
    rec = result["record"]
    _verdict(rec)

    part = rec["part"]; proc = rec["process"]; sim = rec["simulation"]
    fea = rec["distortion_fea"]; dfam = rec["manufacturability"]; insp = rec["inspection_plan"]
    st.markdown("")
    _md(_stat_grid([
        ("Part", part["name"], f"{part['n_facets']:,} facets", True),
        ("Process", proc["name"], proc["material"], False),
        ("Build time", f"{sim['build_time_h']:.2f} h", f"{sim['n_layers']:,} layers", False),
        ("Cost", f"${sim['total_cost_usd']:.2f}", None, False),
        ("Peak warpage", f"{fea['max_distortion_mm']:.3f} mm", "thermal-contraction FEA", False),
        ("DfAM", dfam.get("worst_severity", "ok").upper(), f"{dfam.get('n_critical',0)} critical", False),
    ]))

    down = rec["design_decision"]["chosen_orientation"]
    tightest = insp.get("tightest_tolerance_mm")
    klass, label, _icon = _GATE_META.get(rec["gate"]["decision"], ("rev", rec["gate"]["decision"], "•"))
    films = [
        ("Design", "Orientation chosen",
         f"Rests on {str(down.get('label','')).replace('face ⟂ ','')} · "
         f"{down.get('support_volume_mm3',0):.0f} mm³ support"),
        ("Build", "Simulated",
         f"{sim['n_layers']:,} layers · {sim['build_time_h']:.2f} h · ${sim['total_cost_usd']:.2f}"),
        ("Inspect", "Checked",
         f"DfAM {dfam.get('worst_severity','ok')} · tightest "
         + (f"±{tightest} mm" if tightest is not None else "n/a")),
        ("Decide", "Gated", label),
    ]
    cards = "".join(
        f'<div class="film"><div class="fp">{_esc(p)}</div><div class="fh">{_esc(h)}</div>'
        f'<div class="fl">{_esc(l)}</div></div>' for p, h, l in films)
    st.markdown("")
    _md(f'<div class="filmstrip aba-anim">{cards}</div>')
    st.markdown("")
    _md('<div class="small-note">Use the ribbon above or <b>Start the walkthrough</b> to step through each '
        'stage. Every value here is produced live by the pipeline.</div>')
    _downloads(rec, result)


# --------------------------------------------------------------------------- #
# Navigation ribbon + workspace (fragment => smooth, flicker-free stepping)
# --------------------------------------------------------------------------- #
def _goto(step):
    st.session_state["step"] = step
    st.rerun(scope="fragment")


def _ribbon(step):
    """Phase-grouped ribbon of native buttons. step 0 = overview, 1..7 = stages."""
    tags = "".join(
        f'<span class="phase-tag" style="border-color:{c}55;color:{c}">{_esc(name)}</span>'
        for name, c in _PHASES)
    _md(f'<div class="phase-row aba-anim">{tags}</div>')

    cols = st.columns([1.1, 1, 1, 1, 1, 1, 1, 1])
    if cols[0].button("⌂ Overview", use_container_width=True,
                      type="primary" if step == 0 else "secondary", key="rb_ov"):
        _goto(0)
    for idx, meta in enumerate(_STAGES):
        stage_step = idx + 1
        label = f"{idx+1} · {meta['short']}"
        if cols[idx + 1].button(label, use_container_width=True,
                                type="primary" if step == stage_step else "secondary",
                                key=f"rb_{idx}",
                                help=f"{_PHASES[meta['phase']][0]} — {meta['title']}"):
            _goto(stage_step)


def _stage_header(idx):
    meta = _STAGES[idx]
    phase_name, phase_color = _PHASES[meta["phase"]]
    _md(f'<div class="aba-anim" style="margin:8px 0 2px">'
        f'<span class="aba-eyebrow" style="color:{phase_color}">{_esc(phase_name)} · '
        f'Stage {idx+1} of {len(_STAGES)}</span>'
        f'<div class="aba-h">{_esc(meta["title"])}</div></div>')
    _md(_callout(meta["what"], meta["why"]))


@st.fragment
def _workspace():
    run = st.session_state["run"]
    result, figs, decim = run["result"], run["figs"], run["decim"]
    step = st.session_state.get("step", 0)

    _ribbon(step)

    # top nav: Back | progress | Next
    nl, nm, nr = st.columns([1, 3, 1])
    with nl:
        if st.button("◀ Back", disabled=step == 0, use_container_width=True, key="nav_back"):
            _goto(step - 1)
    with nr:
        if st.button("Next ▶", disabled=step == len(_STAGES), use_container_width=True, key="nav_next"):
            _goto(step + 1)
    with nm:
        if step == 0:
            st.progress(0.0, text="Overview — the whole result at a glance")
        else:
            st.progress(step / len(_STAGES), text=f"Stage {step} of {len(_STAGES)}")

    st.divider()

    body = st.container()
    with body:
        if step == 0:
            _md('<div class="aba-eyebrow aba-anim" style="color:var(--muted)">Result overview</div>')
            _render_overview(result, figs, decim)
        else:
            idx = step - 1
            _stage_header(idx)
            _RENDER[idx](result, figs, decim)

    # bottom continue button for a natural forward flow
    if step < len(_STAGES):
        st.markdown("")
        nxt_label = "Start the walkthrough  ▶" if step == 0 else f"Continue → {_STAGES[step]['short']}"
        bl, bc, br = st.columns([2, 1, 2])
        if bc.button(nxt_label, type="primary", use_container_width=True, key="cont_btn"):
            _goto(step + 1)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
def _sidebar():
    with st.sidebar:
        st.markdown("### Part & process")
        src = st.radio("Input", ["Use a sample part", "Upload an STL"], index=0, key="src")
        uploaded, sample = None, None
        if src == "Upload an STL":
            uploaded = st.file_uploader("STL file (binary or ASCII)", type=["stl"], key="uploader")
        else:
            sample = st.selectbox("Sample part", list(_SAMPLES), index=0, key="sample")
            st.caption(_SAMPLES[sample][2])

        profiles = list_profiles()
        pkey = st.selectbox("Process", [p.key for p in profiles], index=0,
                            format_func=lambda k: get_profile(k).name, key="proc")
        fam = get_profile(pkey).family
        if fam == "FFF":
            st.caption("FFF is the home process — the warpage FEA is calibrated for polymer cooling here.")
        else:
            st.caption(f"{fam} runs natively too; it is shown for cross-process comparison against FFF.")

        grid_n = st.select_slider("Voxel resolution", options=_GRID_OPTIONS, value=64, key="grid_n")
        if grid_n >= 96:
            st.caption("⚠️ Highest resolution — most accurate, but the slowest run.")

        st.button("Run the advisor  ▶", type="primary", use_container_width=True, key="run_btn")
        st.markdown("---")
        with st.expander("What this is"):
            st.markdown(
                "A compact **teaching prototype** of a design-to-inspection *digital thread* for additive "
                "manufacturing. It walks a part through geometry, orientation, a build simulation, a "
                "thermal-contraction **warpage FEA**, DfAM checks, an inspection plan, and a **release gate** "
                "— running the same Python pipeline as the command-line tool. Reduced-order estimates plus a "
                "real linear-elastic FEA with representative process constants; not a production build processor."
            )
    return src, uploaded, sample, pkey, grid_n


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
_md(_CSS)
_md('<div class="aba-hero"><h1>Additive Build Advisor</h1>'
    '<p>Drop in an STL, choose a process, and walk the part through the whole '
    '<b>design → build → inspect → decide</b> thread — orientation, a layer-by-layer build simulation, '
    'the thermal-contraction warpage FEA, manufacturability, and the release gate. Every result on this '
    'page is produced live by the pipeline.</p>'
    '<div class="aba-badges"><span class="aba-badge">FFF home process · SLA / SLS comparison</span>'
    '<span class="aba-badge">scikit-fem warpage FEA</span>'
    '<span class="aba-badge">prototype · teaching demo</span></div></div>')

src, uploaded, sample, pkey, grid_n = _sidebar()

if st.session_state.get("run_btn"):
    try:
        if src == "Upload an STL":
            if uploaded is None:
                st.warning("Choose an STL file in the sidebar first.")
                st.stop()
            data = uploaded.getvalue()
            size_mb = len(data) / (1024 * 1024)
            if len(data) == 0:
                st.error("That STL file is empty. Try another file.")
                st.stop()
            if size_mb > _MAX_UPLOAD_MB:
                st.error(f"That file is {size_mb:.0f} MB — over the {_MAX_UPLOAD_MB} MB limit for this demo. "
                         "Please decimate the mesh or use a smaller part.")
                st.stop()
            tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
            tmp.write(data); tmp.flush(); tmp.close()
            # cheap facet-count guard before the (pure-Python) voxelizer runs
            try:
                from abadvisor.stl_io import read_stl
                tris, _ = read_stl(tmp.name)
            except Exception as e:
                st.error(f"That file could not be read as an STL: {e}")
                st.stop()
            nfac = int(tris.shape[0])
            if nfac < 4:
                st.error("That STL has too few facets to be a solid part.")
                st.stop()
            if nfac > _MAX_FACETS:
                st.error(f"That mesh has {nfac:,} facets — over the {_MAX_FACETS:,} limit for this demo "
                         "(the voxelizer is pure Python). Decimate it and try again.")
                st.stop()
            if nfac > _WARN_FACETS:
                st.info(f"Large mesh ({nfac:,} facets) — this run may take a little longer.")
            key = hashlib.md5(data + pkey.encode() + str(grid_n).encode()).hexdigest()
            with st.spinner("Running the pipeline (parse → orient → voxelize → simulate → FEA → DfAM → gate)…"):
                result, figs, report_html, decim = _run(
                    stl_path=tmp.name, source_label=uploaded.name, process=pkey, grid_n=grid_n, tol=None)
        else:
            factory, tolpath, _desc = _SAMPLES[sample]
            key = f"{sample}-{pkey}-{grid_n}"
            with st.spinner("Running the pipeline…"):
                result, figs, report_html, decim = _run(
                    mesh=factory(), source_label=sample, process=pkey, grid_n=grid_n,
                    tol=_load_tolerances(tolpath))
        st.session_state["run"] = {"result": result, "figs": figs, "report_html": report_html,
                                   "decim": decim, "key": key}
        st.session_state["step"] = 0
        # reset range-dependent stage widgets so a stored index can't exceed the
        # new part's range (grids differ in layer count / candidate count).
        for _k in ("build_layer", "warp_exagg", "orient_sel"):
            st.session_state.pop(_k, None)
    except Exception as e:  # never show a raw stack trace / blank page
        st.session_state.pop("run", None)
        st.error(f"Could not process that part: {e}")

run = st.session_state.get("run")
if not run:
    _md('<div class="aba-card aba-anim" style="text-align:center; padding:34px">'
        '<div style="font-size:40px">🧩</div>'
        '<div class="aba-h" style="margin-top:6px">Pick a part to begin</div>'
        '<div class="tx" style="color:var(--ink2); max-width:640px; margin:6px auto 0">'
        'In the sidebar, choose a sample part (or upload an STL), pick a process, and press '
        '<b>Run the advisor</b>. You will land on an overview, then step through the seven stages of the '
        'build decision.</div></div>')
    st.stop()

_workspace()
